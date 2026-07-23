"""variantes.py — Compara variantes A, B, C, D, E, F.

Desenho experimental (grid 2x3):
  Input:         Close bruto | Log-returns | Close DWT denoised
  Normalização:  Z-score     | Min-Max

  A: Close bruto      + Z-score    (baseline)
  B: Close DWT den.   + Z-score
  C: Log-returns      + Z-score
  D: Log-ret DWT den. + Z-score
  E: Close bruto      + Min-Max
  F: Close DWT den.   + Min-Max

CONTROLE DE VAZAMENTO:
  - Todos os parâmetros de normalização calculados SOMENTE no treino
  - DWT sempre com mode='zero' (causal) — sem espelhamento futuro
  - Threshold de denoising estimado apenas nos coeficientes do treino
  - Target sempre calculado do preço ORIGINAL (nunca do sinal denoised)
"""
from __future__ import annotations

import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Garante que o diretório src está no path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lstm_base import (
    BASE_DIR, DATA_DIR, RESULTS_DIR, TRAIN_FIM,
    FEATURE_COLS, WINNING_CFG,
    log, carregar_clean,
    preparar_features_de,
    engenhar_features, criar_targets,
    wavelet_denoise_series,
    metrics_cls, mcnemar_vs_majority,
    exportar_predicoes, gerar_graficos,
    treinar_em,
)

# ----------------------------------------------------------------- Variante C
def preparar_variant_C(df_clean: pd.DataFrame, detrend_window: int = 252
                       ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Variante C: log-returns detrended → price proxy → features → Z-score.

    BUG CORRIGIDO (versão anterior): o price proxy era `100*exp(cumsum(logret))`,
    que é uma identidade matemática — `cumsum(logret)_t = log(P_t/P_0)`, logo
    `price_proxy_t = (100/P_0) * P_t`, ou seja, apenas o preço ORIGINAL multiplicado
    por uma constante. Como quase todas as 18 features são razões/log-retornos
    invariantes a escala multiplicativa, o proxy antigo produzia features IDÊNTICAS
    à variante A (bug adicional: High/Low não eram reescalados, corrompendo só o
    hl_range). Resultado: A e C treinavam sobre dados numericamente iguais.

    CORREÇÃO: remove de fato a tendência de longo prazo subtraindo do log-retorno
    diário sua média móvel causal (`detrend_window` dias, usa só o passado) antes
    do cumsum. Isso estacionariza a série (drift local removido) e produz um proxy
    genuinamente diferente do preço bruto. High/Low são reescalados pela mesma razão
    do proxy para manter `hl_range` consistente.
    """
    price_orig = df_clean['Price'].astype(float).to_numpy()
    logret     = np.concatenate([[0.0], np.log(price_orig[1:] / price_orig[:-1])])
    logret     = np.where(np.isfinite(logret), logret, 0.0)

    # Detrending causal: subtrai a média móvel do log-retorno (só passado)
    logret_s    = pd.Series(logret)
    drift       = logret_s.shift(1).rolling(detrend_window,
                                             min_periods=detrend_window // 4).mean()
    drift       = drift.fillna(0.0).to_numpy()
    logret_dt   = logret - drift

    # Price proxy: reconstrução via cumsum do log-retorno SEM drift de longo prazo
    price_proxy = 100.0 * np.exp(np.cumsum(logret_dt))

    # Reescala High/Low pela mesma razão do proxy (mantém hl_range consistente)
    ratio = price_proxy / price_orig

    df_used = df_clean.copy()
    df_used['Price']     = price_proxy
    df_used['Close']     = price_proxy
    df_used['Adj Close'] = price_proxy
    df_used['High']      = df_clean['High'].astype(float).to_numpy() * ratio
    df_used['Low']       = df_clean['Low'].astype(float).to_numpy() * ratio

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

    diag = {'variant': 'C',
            'method': f'price_proxy = 100*exp(cumsum(logret - rolling_mean({detrend_window})))',
            'leakage': 'none — rolling mean usa apenas passado (shift(1))'}
    log(f"[C] dataset após drop NaN: X={X.shape}  y={y.shape}")
    return X, y, diag


# ----------------------------------------------------------------- Variante D
def preparar_variant_D(df_clean: pd.DataFrame
                       ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Variante D: DWT denoised log-returns → price proxy → features → Z-score.

    Combina estacionarização (log-returns) com denoising (DWT mode='zero').
    O denoising é aplicado à série de log-returns — remove componentes de
    alta frequência (ruído diário) antes da reconstrução do price proxy.

    ANTI-LEAKAGE:
      - DWT com mode='zero': bordas preenchidas com zeros (sem reflexão futura)
      - Threshold de denoising estimado SOMENTE nos coeficientes do período treino
      - Target calculado do preço ORIGINAL (não do sinal denoised)
    """
    price_orig = df_clean['Price'].astype(float).to_numpy()
    train_mask = (df_clean['Date'] <= TRAIN_FIM).to_numpy()

    # 1. Log-returns causais (logret[0] = 0 para não perder o primeiro ponto)
    logret = np.concatenate([[0.0], np.log(price_orig[1:] / price_orig[:-1])])
    logret = np.where(np.isfinite(logret), logret, 0.0)

    # 2. DWT denoising nos log-returns (mode='zero', threshold do treino)
    logret_den, _ = wavelet_denoise_series(logret, train_mask)

    # 3. Price proxy: reconstrução via cumsum dos log-returns denoised
    price_proxy = 100.0 * np.exp(np.cumsum(logret_den))

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

    diag = {'variant': 'D',
            'method': 'DWT denoised log-returns (mode=zero) + cumsum price proxy',
            'leakage': 'none — mode=zero causal, threshold from train only'}
    log(f"[D] dataset após drop NaN: X={X.shape}  y={y.shape}")
    return X, y, diag


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
    log("VARIANTES — Comparativo A / B / C / D / E / F")
    log("Grid: Input(Close|LogRet|DWT) × Normalização(Z-score|MinMax)")
    log("=" * 60)

    df = carregar_clean()
    log(f"Dataset: {df.shape}  {df['Date'].min().date()} → {df['Date'].max().date()}")

    resultados: list[dict] = []

    # ---- Variante A: Close bruto + Z-score (baseline)
    log("\n--- Variante A: Close bruto + Z-score (baseline) ---")
    XA, yA, _ = preparar_features_de(df, denoise=False, label='A')
    rA = treinar_em(XA, yA, deepcopy(WINNING_CFG), label='A', normalizer='zscore')
    resultados.append(rA)
    exportar_predicoes(rA, df, label='A')
    gerar_graficos(rA, pd.read_csv(RESULTS_DIR / 'predicoes_A.csv',
                                   parse_dates=['Date']), df, label='A')

    # ---- Variante B: Close DWT denoised + Z-score
    log("\n--- Variante B: Close DWT denoised (mode=zero) + Z-score ---")
    XB, yB, _ = preparar_features_de(df, denoise=True, label='B')
    rB = treinar_em(XB, yB, deepcopy(WINNING_CFG), label='B', normalizer='zscore')
    resultados.append(rB)
    exportar_predicoes(rB, df, label='B')
    gerar_graficos(rB, pd.read_csv(RESULTS_DIR / 'predicoes_B.csv',
                                   parse_dates=['Date']), df, label='B')

    # ---- Variante C: Log-returns + Z-score
    log("\n--- Variante C: Log-returns price proxy + Z-score ---")
    XC, yC, _ = preparar_variant_C(df)
    rC = treinar_em(XC, yC, deepcopy(WINNING_CFG), label='C', normalizer='zscore')
    resultados.append(rC)
    exportar_predicoes(rC, df, label='C')
    gerar_graficos(rC, pd.read_csv(RESULTS_DIR / 'predicoes_C.csv',
                                   parse_dates=['Date']), df, label='C')

    # ---- Variante D: Log-returns DWT denoised + Z-score
    log("\n--- Variante D: Log-returns DWT denoised (mode=zero) + Z-score ---")
    XD, yD, _ = preparar_variant_D(df)
    rD = treinar_em(XD, yD, deepcopy(WINNING_CFG), label='D', normalizer='zscore')
    resultados.append(rD)
    exportar_predicoes(rD, df, label='D')
    gerar_graficos(rD, pd.read_csv(RESULTS_DIR / 'predicoes_D.csv',
                                   parse_dates=['Date']), df, label='D')

    # ---- Variante E: Close bruto + Min-Max
    log("\n--- Variante E: Close bruto + Min-Max ---")
    XE, yE, _ = preparar_features_de(df, denoise=False, label='E')
    rE = treinar_em(XE, yE, deepcopy(WINNING_CFG), label='E', normalizer='minmax')
    resultados.append(rE)
    exportar_predicoes(rE, df, label='E')
    gerar_graficos(rE, pd.read_csv(RESULTS_DIR / 'predicoes_E.csv',
                                   parse_dates=['Date']), df, label='E')

    # ---- Variante F: Close DWT denoised + Min-Max
    log("\n--- Variante F: Close DWT denoised (mode=zero) + Min-Max ---")
    XF, yF, _ = preparar_features_de(df, denoise=True, label='F')
    rF = treinar_em(XF, yF, deepcopy(WINNING_CFG), label='F', normalizer='minmax')
    resultados.append(rF)
    exportar_predicoes(rF, df, label='F')
    gerar_graficos(rF, pd.read_csv(RESULTS_DIR / 'predicoes_F.csv',
                                   parse_dates=['Date']), df, label='F')

    # ---- Salva sumário e gráficos comparativos
    log("\n--- Salvando sumário e gráficos comparativos ---")
    salvar_sumario(resultados)
    gerar_plot_comparativo(resultados)
    gerar_plot_probabilidades_comparativo(resultados)
    gerar_plot_overfitting(resultados)

    log("\n=== CONCLUÍDO ===")


if __name__ == '__main__':
    main()
