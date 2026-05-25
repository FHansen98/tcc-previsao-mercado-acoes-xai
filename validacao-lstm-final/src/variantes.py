"""variantes.py — Compara variantes A, B, C, D_corrigida, E, F, G.

Cada variante testa uma combinação diferente de pré-processamento de dados
(denoising wavelet, normalização) para previsão direcional do S&P500.

CONTROLE DE VAZAMENTO:
  - Todos os parâmetros de normalização são calculados SOMENTE no treino
  - DWT sempre com mode='zero' (causal)
  - MODWT causal: cada ponto t usa apenas [t-256, t]
  - Target sempre calculado do preço ORIGINAL
"""
from __future__ import annotations

import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Garante que o diretório src está no path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lstm_base import (
    BASE_DIR, DATA_DIR, RESULTS_DIR, IND, WAVELET, WAVELET_MODE,
    THRESHOLD_MODE, TRAIN_FIM, VAL_INI, TEST_INI,
    FEATURE_COLS, WINNING_CFG, SEED,
    log, carregar_clean,
    preparar_features_de, preparar_variant_D_causal,
    standardize_train, minmax_train, rolling_zscore,
    engenhar_features, criar_targets,
    wavelet_denoise_series, _modwt_causal_denoise_series,
    make_seq, split_by_date, build_model,
    metrics_cls, mcnemar_vs_majority,
    exportar_predicoes, gerar_graficos,
    treinar_em,
)

# ----------------------------------------------------------------- Variante C
def preparar_variant_C(df_clean: pd.DataFrame
                       ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Variante C: log-returns → price proxy (detrended) → features → Z-score.

    Remove a tendência de longo prazo substituindo o preço absoluto por um
    price proxy reconstruído via cumsum de log-returns.  100% causal: usa
    apenas preços observados até cada ponto t.
    """
    price_orig = df_clean['Price'].astype(float).to_numpy()
    logret     = np.concatenate([[0.0], np.log(price_orig[1:] / price_orig[:-1])])
    logret     = np.where(np.isfinite(logret), logret, 0.0)

    # Price proxy: começa em 100 e cresce pelos log-returns reais (sem denoising)
    price_proxy = 100.0 * np.exp(np.cumsum(logret))

    df_used = df_clean.copy()
    df_used['Price']     = price_proxy
    df_used['Close']     = price_proxy
    df_used['Adj Close'] = price_proxy

    # Target do preço ORIGINAL
    price_s = pd.Series(price_orig)
    df_used['logret_1d_orig'] = np.log(price_s / price_s.shift(1)).to_numpy()

    df_feat = engenhar_features(df_used)
    df_feat['logret_1d_orig'] = df_used['logret_1d_orig']
    df_feat = criar_targets(df_feat)

    mask = (df_feat[FEATURE_COLS].notna().all(axis=1) &
            df_feat['target_direction_t+1'].notna())
    X = df_feat.loc[mask, ['Date'] + FEATURE_COLS].reset_index(drop=True)
    y = df_feat.loc[mask, ['Date', 'target_direction_t+1']].reset_index(drop=True)

    diag = {'variant': 'C', 'method': 'price_proxy = 100*exp(cumsum(logret))',
            'leakage': 'none — causal cumsum'}
    log(f"[C] dataset após drop NaN: X={X.shape}  y={y.shape}")
    return X, y, diag


# ----------------------------------------------------------------- Variante G
def preparar_variant_G(df_clean: pd.DataFrame
                       ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Variante G: MODWT causal no Close → features → Min-Max.

    Diferença vs. Variante B: denoising rolling (causal) em vez de DWT global.
    Diferença vs. Variante D: denoising aplicado no Close diretamente (não nos
    log-returns), sem cumsum.  Min-Max em vez de Z-score.
    """
    train_mask = (df_clean['Date'] <= TRAIN_FIM).to_numpy()
    close_orig = df_clean['Close'].astype(float).to_numpy()

    close_den = _modwt_causal_denoise_series(close_orig, train_mask)

    df_used = df_clean.copy()
    df_used['Price']     = close_den
    df_used['Close']     = close_den
    df_used['Adj Close'] = close_den

    price_s = pd.Series(close_orig)
    df_used['logret_1d_orig'] = np.log(price_s / price_s.shift(1)).to_numpy()

    df_feat = engenhar_features(df_used)
    df_feat['logret_1d_orig'] = df_used['logret_1d_orig']
    df_feat = criar_targets(df_feat)

    mask = (df_feat[FEATURE_COLS].notna().all(axis=1) &
            df_feat['target_direction_t+1'].notna())
    X = df_feat.loc[mask, ['Date'] + FEATURE_COLS].reset_index(drop=True)
    y = df_feat.loc[mask, ['Date', 'target_direction_t+1']].reset_index(drop=True)

    diag = {'variant': 'G', 'method': 'MODWT causal on Close (win=256)',
            'leakage': 'none — rolling causal'}
    log(f"[G] dataset após drop NaN: X={X.shape}  y={y.shape}")
    return X, y, diag


# ----------------------------------------------------------------- Variante F
# Colunas extras que serão adicionadas pelas sub-bandas DWT
FILTER_BANK_COLS = ['fb_approx', 'fb_detail1', 'fb_detail2']
ALL_FEATURE_COLS_F = FEATURE_COLS + FILTER_BANK_COLS


def _dwt_filter_bank(price: np.ndarray, train_mask: np.ndarray,
                     wavelet: str = WAVELET, level: int = 3
                     ) -> dict[str, np.ndarray]:
    """Decompõe o preço em 3 níveis DWT e reconstrói cada sub-banda em domínio temporal.

    Todas as sub-bandas são reconstruídas de volta para N pontos usando waverec.
    Sigma e threshold estimados SOMENTE no treino.
    Modo 'zero' (causal).
    """
    n      = len(price)
    coeffs = pywt.wavedec(price, wavelet, mode=WAVELET_MODE, level=level)
    cA     = coeffs[0]
    cD     = coeffs[1:]  # cD[0] = finest (level 1), ..., cD[-1] = coarsest

    # Estimativa do threshold (apenas treino, usando detalhe mais fino)
    finest     = cD[-1]
    nd         = len(finest)
    pos_orig   = np.linspace(0, n - 1, nd).astype(int)
    train_det  = train_mask[pos_orig]
    if train_det.sum() < 8:
        train_det = np.ones_like(train_det, dtype=bool)
    sigma     = np.median(np.abs(finest[train_det])) / 0.6745
    threshold = sigma * np.sqrt(2.0 * np.log(max(n, 2)))

    # Reconstrução: aproximação (cA com zeros para detalhes)
    zeros   = [np.zeros_like(cd) for cd in cD]
    approx  = pywt.waverec([cA] + zeros, wavelet, mode=WAVELET_MODE)[:n]

    # Reconstrução detalhe nível 1 (mais fino)
    coeff1  = [np.zeros_like(cA)] + zeros[:]
    coeff1[1] = pywt.threshold(cD[0], threshold, mode=THRESHOLD_MODE)
    detail1 = pywt.waverec(coeff1, wavelet, mode=WAVELET_MODE)[:n]

    # Reconstrução detalhe nível 2
    coeff2  = [np.zeros_like(cA)] + zeros[:]
    coeff2[2] = pywt.threshold(cD[1], threshold, mode=THRESHOLD_MODE)
    detail2 = pywt.waverec(coeff2, wavelet, mode=WAVELET_MODE)[:n]

    return {'fb_approx': approx, 'fb_detail1': detail1, 'fb_detail2': detail2}


def preparar_variant_F(df_clean: pd.DataFrame
                       ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Variante F: banco de filtros DWT de 3 níveis como canais extras.

    Às 18 features canônicas são adicionadas 3 sub-bandas (approximation, detail1,
    detail2) reconstruídas para o domínio temporal.  Normalização Min-Max por canal
    calculada apenas no treino.
    """
    price_orig = df_clean['Close'].astype(float).to_numpy()
    train_mask = (df_clean['Date'] <= TRAIN_FIM).to_numpy()

    # Calcula as 18 features a partir do preço original (sem denoising)
    price_s = pd.Series(price_orig)
    df_used = df_clean.copy()
    df_used['logret_1d_orig'] = np.log(price_s / price_s.shift(1)).to_numpy()
    df_feat = engenhar_features(df_used)
    df_feat['logret_1d_orig'] = df_used['logret_1d_orig']
    df_feat = criar_targets(df_feat)

    # Adiciona as 3 sub-bandas
    bands = _dwt_filter_bank(price_orig, train_mask, level=3)
    for bname, bvals in bands.items():
        df_feat[bname] = bvals

    mask = (df_feat[FEATURE_COLS].notna().all(axis=1) &
            df_feat['target_direction_t+1'].notna())
    X = df_feat.loc[mask, ['Date'] + ALL_FEATURE_COLS_F].reset_index(drop=True)
    y = df_feat.loc[mask, ['Date', 'target_direction_t+1']].reset_index(drop=True)

    diag = {'variant': 'F',
            'method': 'DWT filter bank level=3 — 3 extra channels (approx, d1, d2)',
            'leakage': 'none — mode=zero, threshold from train only'}
    log(f"[F] dataset após drop NaN: X={X.shape}  y={y.shape}")
    return X, y, diag


def treinar_variant_F(X: pd.DataFrame, y: pd.DataFrame, cfg: dict,
                      label: str) -> dict:
    """Versão de treinar_em que usa ALL_FEATURE_COLS_F (18+3 colunas)."""
    import random, tensorflow as tf
    from tensorflow import keras
    from sklearn.metrics import roc_auc_score

    def reset():
        random.seed(SEED); np.random.seed(SEED)
        tf.random.set_seed(SEED); keras.utils.set_random_seed(SEED)

    # MinMax por canal (apenas treino)
    all_cols = ALL_FEATURE_COLS_F
    train    = X['Date'] <= TRAIN_FIM
    mn       = X.loc[train, all_cols].min()
    mx       = X.loc[train, all_cols].max()
    rng      = (mx - mn).replace(0, 1.0)
    Xs       = X.copy()
    Xs[all_cols] = (X[all_cols] - mn) / rng

    # Sequências
    Xa  = Xs[all_cols].to_numpy(dtype=np.float32)
    ya  = y['target_direction_t+1'].to_numpy(dtype=np.int8)
    ds  = pd.to_datetime(Xs['Date'].to_numpy())
    lb  = cfg['lookback']
    Xl, yl, dl = [], [], []
    for i in range(lb - 1, len(Xa)):
        Xl.append(Xa[i - lb + 1:i + 1])
        yl.append(ya[i])
        dl.append(ds[i])
    Xall = np.array(Xl, dtype=np.float32)
    yall = np.array(yl, dtype=np.int8)
    dall = np.array(dl)

    tr = dall < VAL_INI
    va = (dall >= VAL_INI) & (dall < TEST_INI)
    te = dall >= TEST_INI
    Xtr, ytr = Xall[tr], yall[tr]
    Xva, yva = Xall[va], yall[va]
    Xte, yte = Xall[te], yall[te]
    log(f"[{label}] shapes train/val/test = {Xtr.shape}/{Xva.shape}/{Xte.shape}")

    reset()
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
        'label': label, 'normalizer': 'minmax_per_channel',
        'train': metrics_cls(ytr, pd_tr, pr_tr),
        'val':   metrics_cls(yva, pd_va, pr_va),
        'test':  metrics_cls(yte, pd_te, pr_te),
        'mcnemar_test':   mcnemar_vs_majority(yte, pd_te),
        'epochs_trained': len(h.history['loss']),
        'best_val_epoch': int(np.argmin(h.history['val_loss'])),
        'best_val_loss':  float(min(h.history['val_loss'])),
        'time_s': round(dt, 1),
        'history': {
            'loss':     [float(v) for v in h.history['loss']],
            'val_loss': [float(v) for v in h.history['val_loss']],
        },
        'shapes': {'train': list(Xtr.shape), 'val': list(Xva.shape),
                   'test': list(Xte.shape)},
        'y_proba_test': pr_te.tolist(),
        'y_true_test':  yte.tolist(),
        'scaler': {'type': 'minmax_per_channel', 'cols': all_cols},
    }


# ----------------------------------------------------------------- Sumário comparativo
def salvar_sumario(resultados: list[dict]) -> None:
    """Salva JSON completo e CSV resumido com todas as variantes."""
    path_json = RESULTS_DIR / 'variantes_sp500.json'
    path_csv  = RESULTS_DIR / 'variantes_sp500.csv'

    with open(path_json, 'w') as f:
        json.dump(resultados, f, indent=2, default=str)
    log(f"JSON completo salvo em {path_json}")

    rows = []
    for r in resultados:
        tr, va, te = r['train'], r['val'], r['test']
        rows.append({
            'variante':        r['label'],
            'normalizer':      r.get('normalizer', '—'),
            'acc_train':       round(tr['accuracy'], 4),
            'acc_val':         round(va['accuracy'], 4),
            'acc_test':        round(te['accuracy'], 4),
            'f1_test':         round(te['f1'], 4),
            'mcc_test':        round(te['matthews'], 4),
            'auc_test':        round(te.get('roc_auc', float('nan')), 4),
            'epochs_trained':  r['epochs_trained'],
            'best_val_loss':   round(r['best_val_loss'], 5),
            'mcnemar_p':       round(r['mcnemar_test']['p_value'], 4),
            'time_s':          r['time_s'],
        })
    df = pd.DataFrame(rows).sort_values('mcc_test', ascending=False)
    df.to_csv(path_csv, index=False)
    log(f"CSV resumido salvo em {path_csv}")

    log("\n===== TABELA COMPARATIVA (ordenada por MCC) =====")
    log(df.to_string(index=False))


def gerar_plot_comparativo(resultados: list[dict]) -> None:
    """Gera gráfico comparativo de barras com métricas de todas as variantes."""
    plots_dir = RESULTS_DIR / 'plots'
    plots_dir.mkdir(exist_ok=True)

    labels = [r['label'] for r in resultados]
    acc    = [r['test']['accuracy']  for r in resultados]
    f1     = [r['test']['f1']        for r in resultados]
    mcc    = [r['test']['matthews']  for r in resultados]
    auc    = [r['test'].get('roc_auc', float('nan')) for r in resultados]

    x = np.arange(len(labels))
    w = 0.2

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Comparativo de Variantes — Test Set (2023–2024)', fontsize=13)

    # Barras: Acc, F1, AUC
    ax = axes[0]
    ax.bar(x - w, acc, w, label='Accuracy', color='steelblue')
    ax.bar(x,     f1,  w, label='F1-Score', color='seagreen')
    ax.bar(x + w, auc, w, label='AUC-ROC',  color='darkorange')
    ax.axhline(0.5, color='red', ls='--', lw=0.8, label='Baseline (0.50)')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylabel('Score'); ax.set_ylim(0, 1)
    ax.set_title('Accuracy / F1 / AUC (maior = melhor)')
    ax.legend(fontsize=8)

    # MCC separado (pode ser negativo)
    ax = axes[1]
    colors = ['green' if v > 0.05 else ('orange' if v >= 0 else 'red') for v in mcc]
    bars = ax.bar(x, mcc, color=colors, alpha=0.85)
    ax.axhline(0, color='black', lw=0.8, ls='--')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylabel('MCC'); ax.set_title('Matthews Correlation Coefficient\n(0 = chute; +1 = perfeito)')
    for bar, val in zip(bars, mcc):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                f'{val:+.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    path = plots_dir / 'comparativo_variantes.png'
    plt.savefig(path, dpi=130)
    plt.close()
    log(f"Gráfico comparativo salvo em {path}")


def gerar_plot_probabilidades_comparativo(resultados: list[dict]) -> None:
    """Grid de histogramas P(sobe) por variante — diagnóstico de degeneração."""
    plots_dir = RESULTS_DIR / 'plots'
    plots_dir.mkdir(exist_ok=True)

    n   = len(resultados)
    nc  = 4
    nr  = int(np.ceil(n / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(14, 3.5 * nr))
    fig.suptitle('Distribuição P(sobe) por Variante — Test Set', fontsize=12)
    axes = axes.flatten()

    for i, r in enumerate(resultados):
        ax     = axes[i]
        y_true = np.array(r['y_true_test'])
        y_prob = np.array(r['y_proba_test'])
        ax.hist(y_prob[y_true == 0], bins=25, alpha=0.6, color='red',   label='Cai (0)')
        ax.hist(y_prob[y_true == 1], bins=25, alpha=0.6, color='green', label='Sobe (1)')
        ax.axvline(0.5, color='black', ls='--', lw=0.8)
        t = r['test']
        std_p = float(np.std(y_prob))
        ax.set_title(f"{r['label']}\nMCC={t['matthews']:+.3f}  std={std_p:.3f}",
                     fontsize=9)
        ax.legend(fontsize=6); ax.set_xlabel('P(sobe)', fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = plots_dir / 'probabilidades_variantes.png'
    plt.savefig(path, dpi=130)
    plt.close()
    log(f"Gráfico probabilidades salvo em {path}")


def gerar_plot_overfitting(resultados: list[dict]) -> None:
    """Curvas de loss treino/val para todas as variantes — diagnóstico de overfitting."""
    plots_dir = RESULTS_DIR / 'plots'
    plots_dir.mkdir(exist_ok=True)

    n   = len(resultados)
    nc  = 4
    nr  = int(np.ceil(n / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(14, 3.5 * nr))
    fig.suptitle('Loss por Época — Diagnóstico de Overfitting', fontsize=12)
    axes = axes.flatten()

    for i, r in enumerate(resultados):
        ax = axes[i]
        h  = r['history']
        ax.plot(h['loss'],     label='Treino', lw=1.2)
        ax.plot(h['val_loss'], label='Val',    lw=1.2)
        ax.axvline(r['best_val_epoch'], color='red', ls='--', lw=0.7,
                   label=f"best@{r['best_val_epoch']}")
        ax.set_title(f"{r['label']}  ({r['epochs_trained']} épocas)", fontsize=9)
        ax.set_xlabel('Época', fontsize=7); ax.set_ylabel('BCE', fontsize=7)
        ax.legend(fontsize=6)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = plots_dir / 'overfitting_variantes.png'
    plt.savefig(path, dpi=130)
    plt.close()
    log(f"Gráfico overfitting salvo em {path}")


# ----------------------------------------------------------------- Main
def main() -> None:
    log("=" * 60)
    log("VARIANTES — Comparativo A / B / C / D_corrigida / E / F / G")
    log("=" * 60)

    df = carregar_clean()
    log(f"Dataset: {df.shape}  {df['Date'].min().date()} → {df['Date'].max().date()}")

    resultados: list[dict] = []

    # ---- Variante A: baseline (Close bruto + Z-score)
    log("\n--- Variante A: baseline (Close + Z-score) ---")
    XA, yA, _ = preparar_features_de(df, denoise=False, label='A')
    rA = treinar_em(XA, yA, deepcopy(WINNING_CFG), label='A', normalizer='zscore')
    resultados.append(rA)
    exportar_predicoes(rA, df, label='A')
    gerar_graficos(rA, pd.read_csv(RESULTS_DIR / 'predicoes_A.csv',
                                   parse_dates=['Date']), df, label='A')

    # ---- Variante B: DWT denoised Close + Z-score (mode='zero', causal)
    log("\n--- Variante B: DWT denoised Close (mode=zero) + Z-score ---")
    XB, yB, _ = preparar_features_de(df, denoise=True, label='B')
    rB = treinar_em(XB, yB, deepcopy(WINNING_CFG), label='B', normalizer='zscore')
    resultados.append(rB)
    exportar_predicoes(rB, df, label='B')
    gerar_graficos(rB, pd.read_csv(RESULTS_DIR / 'predicoes_B.csv',
                                   parse_dates=['Date']), df, label='B')

    # ---- Variante C: log-returns price proxy + Z-score
    log("\n--- Variante C: log-returns price proxy + Z-score ---")
    XC, yC, _ = preparar_variant_C(df)
    rC = treinar_em(XC, yC, deepcopy(WINNING_CFG), label='C', normalizer='zscore')
    resultados.append(rC)
    exportar_predicoes(rC, df, label='C')
    gerar_graficos(rC, pd.read_csv(RESULTS_DIR / 'predicoes_C.csv',
                                   parse_dates=['Date']), df, label='C')

    # ---- Variante D_corrigida: MODWT causal em log-returns + Z-score
    log("\n--- Variante D_corrigida: MODWT causal log-returns + Z-score ---")
    XD, yD, _ = preparar_variant_D_causal(df)
    rD = treinar_em(XD, yD, deepcopy(WINNING_CFG), label='D_corrigida',
                   normalizer='zscore')
    resultados.append(rD)
    exportar_predicoes(rD, df, label='D_corrigida')
    gerar_graficos(rD, pd.read_csv(RESULTS_DIR / 'predicoes_D_corrigida.csv',
                                   parse_dates=['Date']), df, label='D_corrigida')

    # ---- Variante E: Close + Rolling Z-score causal (252d)
    log("\n--- Variante E: Close + Rolling Z-score (252d) ---")
    XE, yE, _ = preparar_features_de(df, denoise=False, label='E')
    rE = treinar_em(XE, yE, deepcopy(WINNING_CFG), label='E',
                   normalizer='rolling_zscore')
    resultados.append(rE)
    exportar_predicoes(rE, df, label='E')
    gerar_graficos(rE, pd.read_csv(RESULTS_DIR / 'predicoes_E.csv',
                                   parse_dates=['Date']), df, label='E')

    # ---- Variante F: filter bank DWT 3 níveis + MinMax por canal
    log("\n--- Variante F: banco de filtros DWT 3 níveis + MinMax ---")
    XF, yF, _ = preparar_variant_F(df)
    rF = treinar_variant_F(XF, yF, deepcopy(WINNING_CFG), label='F')
    resultados.append(rF)
    exportar_predicoes(rF, df, label='F')
    gerar_graficos(rF, pd.read_csv(RESULTS_DIR / 'predicoes_F.csv',
                                   parse_dates=['Date']), df, label='F')

    # ---- Variante G: MODWT causal Close + MinMax
    log("\n--- Variante G: MODWT causal Close + MinMax ---")
    XG, yG, _ = preparar_variant_G(df)
    rG = treinar_em(XG, yG, deepcopy(WINNING_CFG), label='G', normalizer='minmax')
    resultados.append(rG)
    exportar_predicoes(rG, df, label='G')
    gerar_graficos(rG, pd.read_csv(RESULTS_DIR / 'predicoes_G.csv',
                                   parse_dates=['Date']), df, label='G')

    # ---- Salva sumário e gráficos comparativos
    log("\n--- Salvando sumário e gráficos comparativos ---")
    salvar_sumario(resultados)
    gerar_plot_comparativo(resultados)
    gerar_plot_probabilidades_comparativo(resultados)
    gerar_plot_overfitting(resultados)

    log("\n=== CONCLUÍDO ===")


if __name__ == '__main__':
    main()
