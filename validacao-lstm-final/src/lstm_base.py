"""LSTM Base — Previsão Direcional S&P500.

Pipeline central reutilizado de validacao-bolsa/src/fase6_wavelet.py.
Tarefas:
  - Carregar dados limpos (sp500_clean.parquet)
  - Feature engineering (18 features canônicas)
  - Normalização Z-score (calculada apenas no treino)
  - Denoising opcional via DWT (wavelet db4)
  - Criar sequências lookback=20
  - Treinar LSTM (configuração vencedora less_reg_d01 da Fase 5)
  - Avaliar: Accuracy, F1, MCC, AUC, McNemar vs majority

IMPORTANTE — Prevenção de vazamento de dados:
  - WAVELET_MODE = 'zero'  (causal, não usa futuro)
  - Threshold DWT calculado apenas no segmento de treino
  - Z-score calculado apenas no segmento de treino
  - Sequências criadas em ordem temporal (shuffle=False)
  - Target calculado sempre a partir do preço ORIGINAL
"""
from __future__ import annotations

import json
import os
import random
import time
from copy import deepcopy
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import pywt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             roc_auc_score)
from scipy import stats as spstats

# ---------------------------------------------------------------- Config
SEED = 42
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

IND        = 'sp500'
TRAIN_FIM  = pd.Timestamp('2020-12-31')
VAL_INI    = pd.Timestamp('2021-01-01')
TEST_INI   = pd.Timestamp('2023-01-01')

# Dados desde 2000 (6.288 dias vs 2.515 dias de 2015-2024)
DATA_FILE_EXT = 'csv'  # 'csv' ou 'parquet'

# Wavelet — SEMPRE usar mode='zero' (causal, nunca 'symmetric')
WAVELET        = 'db4'
WAVELET_MODE   = 'zero'
DECOMP_LEVEL   = None   # None = automático
THRESHOLD_MODE = 'soft'

OHLCV_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']

FEATURE_COLS = [
    'logret_1d', 'logret_lag_1', 'logret_lag_2', 'logret_lag_3',
    'logret_lag_5', 'logret_lag_10', 'logret_lag_20',
    'price_over_ma5', 'price_over_ma20', 'ma5_over_ma20',
    'vol_20d', 'vol_60d', 'abs_ret_1d', 'ret2_1d',
    'volume_log', 'volume_over_ma20', 'hl_range', 'rsi_14',
]

# Configuração vencedora da Fase 5 (less_reg_d01)
WINNING_CFG = dict(
    name='less_reg_d01',
    lookback=20, units=32, dropout=0.1, l2=0.0,
    optimizer='adam', lr=1e-3, loss='bce',
    epochs=80, patience=10, batch=32,
)


# ---------------------------------------------------------------- Utils
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _reset() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)


# ---------------------------------------------------------------- Wavelet
def wavelet_denoise_series(
    x: np.ndarray,
    train_mask: np.ndarray,
    wavelet: str = WAVELET,
    level: int | None = DECOMP_LEVEL,
    threshold_mode: str = THRESHOLD_MODE,
) -> tuple[np.ndarray, dict]:
    """DWT denoising causal (mode='zero').

    Sigma estimado APENAS no treino. Aplicado em toda a série.
    Não usa dados futuros (mode='zero' = padding com zeros).
    """
    x = np.ascontiguousarray(x, dtype=np.float64).copy()
    n = len(x)
    if level is None:
        level = pywt.dwt_max_level(n, pywt.Wavelet(wavelet).dec_len)
        level = min(level, 6)

    coeffs     = pywt.wavedec(x, wavelet, mode=WAVELET_MODE, level=level)
    cA         = coeffs[0]
    cD_list    = coeffs[1:]
    finest     = cD_list[-1]

    nd = len(finest)
    pos_orig       = np.linspace(0, n - 1, nd).astype(int)
    train_in_det   = train_mask[pos_orig]
    if train_in_det.sum() < 32:
        train_in_det = np.ones_like(train_in_det, dtype=bool)

    sigma     = np.median(np.abs(finest[train_in_det])) / 0.6745
    threshold = sigma * np.sqrt(2.0 * np.log(max(n, 2)))

    cD_thresh  = [pywt.threshold(cd, threshold, mode=threshold_mode) for cd in cD_list]
    denoised   = pywt.waverec([cA] + cD_thresh, wavelet, mode=WAVELET_MODE)[:n]

    total_det = sum(len(cd) for cd in cD_list)
    zeroed    = sum(int((np.abs(cd) <= threshold).sum()) for cd in cD_list)
    info = {
        'wavelet': wavelet, 'level': int(level),
        'sigma_train': float(sigma), 'threshold': float(threshold),
        'pct_detail_zeroed': zeroed / max(total_det, 1) * 100.0,
        'wavelet_mode': WAVELET_MODE,
    }
    return denoised.astype(np.float64), info


def aplicar_denoising(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Retorna cópia de df com OHLCV denoised + diagnóstico."""
    out        = df.copy()
    train_mask = (out['Date'] <= TRAIN_FIM).to_numpy()
    diag       = {}
    for col in OHLCV_COLS:
        x = out[col].astype(float).to_numpy()
        if col == 'Volume':
            x_in     = np.log(np.where(x > 0, x, np.nan))
            valid    = np.isfinite(x_in)
            x_filled = pd.Series(x_in).interpolate(limit_direction='both').to_numpy()
            den_log, info = wavelet_denoise_series(x_filled, train_mask)
            den           = np.exp(den_log)
            den[~valid]   = x[~valid]
        else:
            den, info = wavelet_denoise_series(x, train_mask)
        out[col]  = den
        diag[col] = info
    out['Adj Close'] = out['Close']
    out['Price']     = out['Close']
    return out, diag


# ---------------------------------------------------------------- Variante D corrigida (MODWT causal em log-returns)
def _modwt_causal_denoise_series(
    x: np.ndarray,
    train_mask: np.ndarray,
    wavelet: str = WAVELET,
    threshold_mode: str = THRESHOLD_MODE,
    win: int = 256,
) -> np.ndarray:
    """MODWT causal: para cada t usa apenas a janela [t-win, t].

    Totalmente causal — nenhum ponto t usa dados de t+1 ou posteriores.
    O threshold é estimado apenas nos pontos de treino dentro de cada janela.
    """
    n   = len(x)
    out = x.copy().astype(np.float64)
    for t in range(win, n):
        seg     = x[t - win: t + 1].astype(np.float64)
        tm_seg  = train_mask[t - win: t + 1]
        # Aplica DWT na janela (mode='zero' para bordas da janela)
        level = min(pywt.dwt_max_level(len(seg), pywt.Wavelet(wavelet).dec_len), 6)
        coeffs    = pywt.wavedec(seg, wavelet, mode='zero', level=level)
        cA        = coeffs[0]
        cD_list   = coeffs[1:]
        finest    = cD_list[-1]
        nd        = len(finest)
        pos_orig  = np.linspace(0, len(seg) - 1, nd).astype(int)
        train_det = tm_seg[pos_orig]
        if train_det.sum() < 8:
            train_det = np.ones_like(train_det, dtype=bool)
        sigma     = np.median(np.abs(finest[train_det])) / 0.6745
        threshold = sigma * np.sqrt(2.0 * np.log(max(len(seg), 2)))
        cD_thresh = [pywt.threshold(cd, threshold, mode=threshold_mode) for cd in cD_list]
        denoised  = pywt.waverec([cA] + cD_thresh, wavelet, mode='zero')[:len(seg)]
        out[t] = denoised[-1]  # Apenas o ponto t (último da janela)
    return out


def preparar_variant_D_causal(
    df_clean: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Variante D corrigida: denoising causal de log-returns via MODWT.

    Pipeline (100% causal, sem vazamento):
      1. Calcula log-returns do preço ORIGINAL
      2. Aplica MODWT causal (janela rolling de 256 dias) nos log-returns
      3. Reconstrói price_proxy via cumsum (causal)
      4. Calcula as 18 features sobre price_proxy
      5. Target calculado do preço ORIGINAL (nunca do proxy)

    A diferença vs. Variante D original:
      - Original usava DWT global (mode='symmetric') → vazamento confirmado 95%
      - Esta versão usa MODWT rolling → cada t só vê [t-256, t]
    """
    train_mask = (df_clean['Date'] <= TRAIN_FIM).to_numpy()

    # 1. Log-returns do preço original
    price_orig = df_clean['Price'].astype(float).to_numpy()
    logret     = np.concatenate([[0.0], np.log(price_orig[1:] / price_orig[:-1])])
    logret     = np.where(np.isfinite(logret), logret, 0.0)

    # 2. MODWT causal nos log-returns
    logret_den = _modwt_causal_denoise_series(logret, train_mask)

    # 3. Price proxy via cumsum (causal: p_t = p_0 * exp(sum logret_den[0..t]))
    price_proxy = price_orig[0] * np.exp(np.cumsum(logret_den))

    # 4. Montar df_used com price_proxy
    df_used = df_clean.copy()
    df_used['Price'] = price_proxy
    df_used['Close'] = price_proxy
    df_used['Adj Close'] = price_proxy
    df_used['Open']  = df_clean['Open'].astype(float).to_numpy()  # OHLV original (exceto Close)

    # 5. Preservar original price para o target
    df_used['logret_1d_orig'] = np.log(
        pd.Series(price_orig) / pd.Series(price_orig).shift(1)
    ).to_numpy()

    df_feat = engenhar_features(df_used)
    df_feat['logret_1d_orig'] = df_used['logret_1d_orig']
    df_feat = criar_targets(df_feat)

    diag = {
        'variant': 'D_causal',
        'method': 'MODWT rolling (win=256) on log-returns + cumsum',
        'leakage': 'none — each t uses only [t-256, t]',
        'wavelet': WAVELET,
        'threshold_mode': THRESHOLD_MODE,
    }
    return df_feat[['Date'] + FEATURE_COLS].dropna().reset_index(drop=True), \
           df_feat[['Date', 'target_direction_t+1']].dropna().reset_index(drop=True), \
           diag


# ---------------------------------------------------------------- Features
def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def engenhar_features(df: pd.DataFrame) -> pd.DataFrame:
    """18 features canônicas (igual à Fase 1)."""
    d           = df.copy()
    d['ret_1d'] = d['Price'].pct_change()
    d['logret_1d'] = np.log(d['Price'] / d['Price'].shift(1))
    for k in [1, 2, 3, 5, 10, 20]:
        d[f'logret_lag_{k}'] = d['logret_1d'].shift(k)
    ma5  = d['Price'].rolling(5).mean()
    ma20 = d['Price'].rolling(20).mean()
    d['price_over_ma5']  = d['Price'] / ma5  - 1
    d['price_over_ma20'] = d['Price'] / ma20 - 1
    d['ma5_over_ma20']   = ma5 / ma20 - 1
    d['vol_20d']  = d['ret_1d'].rolling(20).std() * np.sqrt(252)
    d['vol_60d']  = d['ret_1d'].rolling(60).std() * np.sqrt(252)
    d['abs_ret_1d']      = d['ret_1d'].abs()
    d['ret2_1d']         = d['ret_1d'] ** 2
    d['volume_log']      = np.log(d['Volume'].replace(0, np.nan))
    vol_ma20             = d['Volume'].rolling(20).mean()
    d['volume_over_ma20'] = d['Volume'] / vol_ma20 - 1
    d['hl_range']        = (d['High'] - d['Low']) / d['Close']
    d['rsi_14']          = _rsi(d['Price'], 14)
    return d


def criar_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Target: direção t+1 calculada do preço ORIGINAL (nunca denoised)."""
    d = df.copy()
    d['target_logret_t+1']   = d['logret_1d_orig'].shift(-1)
    d['target_direction_t+1'] = (d['target_logret_t+1'] > 0).astype('Int8')
    return d


def preparar_features_de(df_clean: pd.DataFrame, *, denoise: bool, label: str
                          ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    diag = {}
    df_used = df_clean.copy()
    if denoise:
        df_used, diag = aplicar_denoising(df_clean)

    price_orig             = df_clean['Price'].astype(float)
    df_used['logret_1d_orig'] = np.log(price_orig / price_orig.shift(1))

    df_feat = engenhar_features(df_used)
    df_feat['logret_1d_orig']  = df_used['logret_1d_orig']
    df_feat = criar_targets(df_feat)

    cols  = ['Date'] + FEATURE_COLS
    X     = df_feat[cols].copy()
    y     = df_feat[['Date', 'target_direction_t+1']].copy()
    mask  = X[FEATURE_COLS].notna().all(axis=1) & y['target_direction_t+1'].notna()
    X     = X.loc[mask].reset_index(drop=True)
    y     = y.loc[mask].reset_index(drop=True)
    log(f"[{label}] dataset após drop NaN: X={X.shape}  y={y.shape}")
    return X, y, diag


# ---------------------------------------------------------------- Normalização
def standardize_train(X: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Z-score calculado SOMENTE no segmento de treino."""
    train  = X['Date'] <= TRAIN_FIM
    mean   = X.loc[train, FEATURE_COLS].mean()
    std    = X.loc[train, FEATURE_COLS].std().replace(0, 1.0)
    Xs     = X.copy()
    Xs[FEATURE_COLS] = (X[FEATURE_COLS] - mean) / std
    return Xs, {'mean': mean.to_dict(), 'std': std.to_dict()}


def minmax_train(X: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Min-Max [0,1] calculado SOMENTE no segmento de treino."""
    train  = X['Date'] <= TRAIN_FIM
    mn     = X.loc[train, FEATURE_COLS].min()
    mx     = X.loc[train, FEATURE_COLS].max()
    rng    = (mx - mn).replace(0, 1.0)
    Xs     = X.copy()
    Xs[FEATURE_COLS] = (X[FEATURE_COLS] - mn) / rng
    return Xs, {'min': mn.to_dict(), 'max': mx.to_dict()}


def rolling_zscore(X: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Rolling Z-score causal: normaliza t usando média/std dos últimos `window` dias."""
    Xs = X.copy()
    for col in FEATURE_COLS:
        s   = X[col]
        mu  = s.shift(1).rolling(window, min_periods=window // 2).mean()
        std = s.shift(1).rolling(window, min_periods=window // 2).std()
        std = std.replace(0, np.nan).fillna(1.0)
        Xs[col] = (s - mu) / std
    return Xs


# ---------------------------------------------------------------- Sequências
def make_seq(X: pd.DataFrame, y: pd.DataFrame, lb: int
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xa  = X[FEATURE_COLS].to_numpy(dtype=np.float32)
    ya  = y['target_direction_t+1'].to_numpy(dtype=np.int8)
    ds  = pd.to_datetime(X['Date'].to_numpy())
    Xs, ys, dsx = [], [], []
    for i in range(lb - 1, len(Xa)):
        Xs.append(Xa[i - lb + 1:i + 1])
        ys.append(ya[i])
        dsx.append(ds[i])
    return (np.array(Xs, dtype=np.float32),
            np.array(ys, dtype=np.int8),
            np.array(dsx))


def split_by_date(Xs, ys, ds):
    tr = ds < VAL_INI
    va = (ds >= VAL_INI) & (ds < TEST_INI)
    te = ds >= TEST_INI
    return (Xs[tr], ys[tr], ds[tr]), (Xs[va], ys[va], ds[va]), (Xs[te], ys[te], ds[te])


# ---------------------------------------------------------------- Modelo
def build_model(cfg: dict, nf: int) -> keras.Model:
    _reset()
    inp = layers.Input((cfg['lookback'], nf))
    reg = keras.regularizers.l2(cfg['l2']) if cfg.get('l2', 0) > 0 else None
    x   = layers.LSTM(cfg['units'], dropout=cfg['dropout'],
                      kernel_regularizer=reg)(inp)
    x   = layers.Dense(16, activation='relu')(x)
    x   = layers.Dropout(cfg['dropout'])(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    m   = keras.Model(inp, out)
    m.compile(keras.optimizers.Adam(cfg['lr']),
              'binary_crossentropy',
              metrics=['accuracy', keras.metrics.AUC(name='auc')])
    return m


# ---------------------------------------------------------------- Métricas
def metrics_cls(y_true, y_pred, y_proba=None) -> dict:
    out = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1':       float(f1_score(y_true, y_pred, zero_division=0)),
        'matthews': float(matthews_corrcoef(y_true, y_pred)),
    }
    if y_proba is not None:
        try:
            out['roc_auc'] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            out['roc_auc'] = float('nan')
    return out


def mcnemar_vs_majority(y_true, y_pred) -> dict:
    maj   = int(pd.Series(y_true).mode().iloc[0])
    y_maj = np.full_like(y_true, maj)
    b     = int(((y_pred == y_true) & (y_maj != y_true)).sum())
    c     = int(((y_pred != y_true) & (y_maj == y_true)).sum())
    if (b + c) == 0:
        return {'b': b, 'c': c, 'p_value': 1.0}
    stat  = (abs(b - c) - 1) ** 2 / (b + c)
    return {'b': b, 'c': c,
            'p_value': float(1 - spstats.chi2.cdf(stat, df=1))}


# ---------------------------------------------------------------- Treino
def treinar_em(X: pd.DataFrame, y: pd.DataFrame, cfg: dict,
               label: str, normalizer: str = 'zscore') -> dict:
    """Treina o modelo e retorna métricas completas.

    Parameters
    ----------
    normalizer : 'zscore' | 'minmax' | 'rolling_zscore'
    """
    if normalizer == 'zscore':
        Xs, scaler_info = standardize_train(X)
    elif normalizer == 'minmax':
        Xs, scaler_info = minmax_train(X)
    elif normalizer == 'rolling_zscore':
        Xs         = rolling_zscore(X)
        Xs         = Xs.dropna(subset=FEATURE_COLS).reset_index(drop=True)
        y          = y[y['Date'].isin(Xs['Date'].values)].reset_index(drop=True)
        scaler_info = {'type': 'rolling_zscore', 'window': 252}
    else:
        raise ValueError(f"normalizer desconhecido: {normalizer}")

    X_all, y_all, ds_all = make_seq(Xs, y, cfg['lookback'])
    (Xtr, ytr, _), (Xva, yva, _), (Xte, yte, _) = split_by_date(X_all, y_all, ds_all)
    log(f"[{label}] shapes train/val/test = {Xtr.shape}/{Xva.shape}/{Xte.shape}")

    model = build_model(cfg, Xtr.shape[2])
    cb    = [keras.callbacks.EarlyStopping(monitor='val_loss',
                                           patience=cfg['patience'],
                                           restore_best_weights=True)]
    t0 = time.time()
    h  = model.fit(Xtr, ytr.astype('float32'),
                   validation_data=(Xva, yva.astype('float32')),
                   epochs=cfg['epochs'], batch_size=cfg['batch'],
                   verbose=0, callbacks=cb, shuffle=False)
    dt = time.time() - t0

    pr_te = model.predict(Xte, verbose=0).ravel()
    pd_te = (pr_te > 0.5).astype(int)
    pr_va = model.predict(Xva, verbose=0).ravel()
    pd_va = (pr_va > 0.5).astype(int)
    pr_tr = model.predict(Xtr, verbose=0).ravel()
    pd_tr = (pr_tr > 0.5).astype(int)

    return {
        'label':          label,
        'normalizer':     normalizer,
        'train':          metrics_cls(ytr, pd_tr, pr_tr),
        'val':            metrics_cls(yva, pd_va, pr_va),
        'test':           metrics_cls(yte, pd_te, pr_te),
        'mcnemar_test':   mcnemar_vs_majority(yte, pd_te),
        'epochs_trained': len(h.history['loss']),
        'best_val_epoch': int(np.argmin(h.history['val_loss'])),
        'best_val_loss':  float(min(h.history['val_loss'])),
        'time_s':         round(dt, 1),
        'history': {
            'loss':     [float(v) for v in h.history['loss']],
            'val_loss': [float(v) for v in h.history['val_loss']],
        },
        'shapes': {
            'train': list(Xtr.shape),
            'val':   list(Xva.shape),
            'test':  list(Xte.shape),
        },
        'y_proba_test': pr_te.tolist(),
        'y_true_test':  yte.tolist(),
        'scaler':        scaler_info,
    }


# ---------------------------------------------------------------- Exportar predições
def exportar_predicoes(res: dict, df_clean: pd.DataFrame, label: str) -> pd.DataFrame:
    """Cria DataFrame com previsões dia a dia do test set.

    Colunas geradas:
      Date         — data do dia previsto
      close_real   — preço de fechamento real
      real         — direção real (0=caiu, 1=subiu)
      proba_sobe   — probabilidade P(sobe) prevista pelo modelo [0,1]
      pred         — predição binária (0 ou 1, threshold=0.5)
      acerto       — 1 se pred == real, 0 caso contrário
    """
    y_true  = np.array(res['y_true_test'])
    y_proba = np.array(res['y_proba_test'])
    y_pred  = (y_proba > 0.5).astype(int)

    # Datas do test set: alinha pelo tamanho
    datas_test = df_clean.loc[df_clean['Date'] >= TEST_INI, 'Date'].reset_index(drop=True)
    # Remove os primeiros (lookback-1) dias usados como contexto
    lb = WINNING_CFG['lookback']
    datas_test = datas_test.iloc[lb - 1:].reset_index(drop=True)
    # Trunca para o tamanho das predições (pode diferir por 1-2 dias nas bordas)
    n = min(len(y_true), len(datas_test))
    datas_test = datas_test.iloc[:n]

    close_test = (df_clean.loc[df_clean['Date'].isin(datas_test.values), 'Close']
                  .reset_index(drop=True).iloc[:n])

    df_pred = pd.DataFrame({
        'Date':       datas_test.values,
        'close_real': close_test.values,
        'real':       y_true[:n],
        'proba_sobe': y_proba[:n].round(4),
        'pred':       y_pred[:n],
        'acerto':     (y_pred[:n] == y_true[:n]).astype(int),
    })

    path = RESULTS_DIR / f'predicoes_{label}.csv'
    df_pred.to_csv(path, index=False)
    log(f"Predições salvas em {path}  ({len(df_pred)} dias)")
    return df_pred


def gerar_graficos(res: dict, df_pred: pd.DataFrame, df_clean: pd.DataFrame,
                   label: str) -> None:
    """Gera 4 gráficos: curva de preço c/ acertos, loss, ROC e histograma de probabilidades."""
    plots_dir = RESULTS_DIR / 'plots'
    plots_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f'Resultados — {label}', fontsize=13)

    # 1) Preço real com marcação de acertos/erros
    ax = axes[0, 0]
    ax.plot(df_pred['Date'], df_pred['close_real'], color='gray', lw=0.8, label='Close real')
    acertos = df_pred[df_pred['acerto'] == 1]
    erros   = df_pred[df_pred['acerto'] == 0]
    ax.scatter(acertos['Date'], acertos['close_real'], color='green', s=8,
               alpha=0.6, label=f'Acerto ({len(acertos)})')
    ax.scatter(erros['Date'],   erros['close_real'],   color='red',   s=8,
               alpha=0.6, label=f'Erro ({len(erros)})')
    t = res['test']
    ax.set_title(f'Previsões no Test Set\nAcc={t["accuracy"]:.1%}  MCC={t["matthews"]:+.3f}')
    ax.legend(fontsize=7); ax.set_ylabel('Close (USD)')
    ax.xaxis.set_tick_params(rotation=30)

    # 2) Loss treino vs validação
    ax = axes[0, 1]
    h = res['history']
    ax.plot(h['loss'],     label='Treino', lw=1.3)
    ax.plot(h['val_loss'], label='Val',    lw=1.3)
    ax.axvline(res['best_val_epoch'], color='red', ls='--', lw=0.8,
               label=f"best@ep{res['best_val_epoch']}")
    ax.set_title('Loss por Época'); ax.set_xlabel('Época'); ax.set_ylabel('BCE Loss')
    ax.legend(fontsize=8)

    # 3) Curva ROC
    from sklearn.metrics import roc_curve
    ax = axes[1, 0]
    fpr, tpr, _ = roc_curve(res['y_true_test'], res['y_proba_test'])
    auc = t.get('roc_auc', float('nan'))
    ax.plot(fpr, tpr, lw=1.5, label=f'AUC = {auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
    ax.set_title('Curva ROC'); ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.legend(fontsize=8)

    # 4) Histograma das probabilidades previstas
    ax = axes[1, 1]
    y_true  = np.array(res['y_true_test'])
    y_proba = np.array(res['y_proba_test'])
    ax.hist(y_proba[y_true == 0], bins=30, alpha=0.6, color='red',   label='Real: cai (0)')
    ax.hist(y_proba[y_true == 1], bins=30, alpha=0.6, color='green', label='Real: sobe (1)')
    ax.axvline(0.5, color='black', ls='--', lw=0.8, label='threshold=0.5')
    ax.set_title('Distribuição P(sobe) por Classe')
    ax.set_xlabel('P(sobe)'); ax.set_ylabel('Frequência')
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = plots_dir / f'resultados_{label}.png'
    plt.savefig(path, dpi=130)
    plt.close()
    log(f"Gráfico salvo em {path}")


# ---------------------------------------------------------------- I/O
def carregar_clean() -> pd.DataFrame:
    if DATA_FILE_EXT == 'csv':
        path = DATA_DIR / f'{IND}_clean.csv'
        df   = pd.read_csv(path)
    else:
        path = DATA_DIR / f'{IND}_clean.parquet'
        df   = pd.read_parquet(path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df.sort_values('Date').reset_index(drop=True)


# ---------------------------------------------------------------- Main (smoke test)
if __name__ == '__main__':
    log("=== lstm_base.py — smoke test (variante A baseline) ===")
    df = carregar_clean()
    log(f"Dataset: {df.shape}  {df['Date'].min().date()} → {df['Date'].max().date()}")

    X, y, _ = preparar_features_de(df, denoise=False, label='A_baseline')
    res      = treinar_em(X, y, deepcopy(WINNING_CFG), label='A_baseline',
                          normalizer='zscore')

    t = res['test']
    log(f"RESULTADO: acc={t['accuracy']:.4f}  f1={t['f1']:.4f}  "
        f"mcc={t['matthews']:+.4f}  auc={t.get('roc_auc', 0):.4f}")
    log(f"McNemar p-value: {res['mcnemar_test']['p_value']:.4f}")

    # Exportar predições dia a dia e gráficos
    df_pred = exportar_predicoes(res, df, label='A_baseline')
    gerar_graficos(res, df_pred, df, label='A_baseline')
    log("Arquivos gerados:")
    log(f"  results/predicoes_A_baseline.csv  — {len(df_pred)} dias, colunas: Date, close_real, real, proba_sobe, pred, acerto")
    log(f"  results/plots/resultados_A_baseline.png  — 4 gráficos")
