"""
Gera gráficos do xLSTM-TS a partir do modelo treinado (sem re-treinar).
Carrega xlstm_ts_model.pth e gera plots similares à Fase 5.
"""

import sys
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_SRC = ROOT / "external" / "src"
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(EXTERNAL_SRC))
sys.path.insert(0, str(SRC_DIR))

import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Monkey patches (mesmos do xlstm_ts.py)
import torch.nn as nn
from xlstm import (
    xLSTMBlockStack, xLSTMBlockStackConfig,
    mLSTMBlockConfig, mLSTMLayerConfig,
    sLSTMBlockConfig, sLSTMLayerConfig,
    FeedForwardConfig,
)
from ml.constants import SEQ_LENGTH_XLSTM
import ml.models.xlstm_ts.xlstm_ts_model as _xlstm_model_mod


def _patched_create_xlstm_model(seq_length):
    input_size = 1
    embedding_dim = 64
    output_size = 1
    cfg = xLSTMBlockStackConfig(
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=4, qkv_proj_blocksize=2, num_heads=2
            )
        ),
        slstm_block=sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                backend="vanilla",
                num_heads=2,
                conv1d_kernel_size=2,
                bias_init="powerlaw_blockdependent",
            ),
            feedforward=FeedForwardConfig(proj_factor=1.1, act_fn="gelu"),
        ),
        context_length=seq_length,
        num_blocks=4,
        embedding_dim=embedding_dim,
        slstm_at=[1],
    )
    xlstm_stack = xLSTMBlockStack(cfg).to(DEVICE)
    input_projection = nn.Linear(input_size, embedding_dim).to(DEVICE)
    output_projection = nn.Linear(embedding_dim, output_size).to(DEVICE)
    return xlstm_stack, input_projection, output_projection


_xlstm_model_mod.create_xlstm_model = _patched_create_xlstm_model

import ml.models.xlstm_ts.preprocessing as _xlstm_pre


def _patched_split_data(x, y, dates, set, train_end_date, val_end_date):
    if set == "train":
        mask = (dates < train_end_date)
    elif set == "val":
        mask = (dates >= train_end_date) & (dates < val_end_date)
    elif set == "test":
        mask = (dates >= val_end_date)
    else:
        raise ValueError("Invalid set name")
    x_s = x[mask].to(DEVICE)
    y_s = y[mask].to(DEVICE)
    return x_s, y_s, dates[mask]


_xlstm_pre._split_data = _patched_split_data

# Imports
from sklearn.preprocessing import MinMaxScaler
import pywt
from ml.models.xlstm_ts.preprocessing import create_sequences, split_train_val_test_xlstm
from ml.models.xlstm_ts.training import evaluate_model

# Configuração
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

TRAIN_END_DATE = datetime.datetime.strptime("2020-12-31", "%Y-%m-%d")
VAL_END_DATE = datetime.datetime.strptime("2022-12-31", "%Y-%m-%d")

TRAIN_FIM = pd.Timestamp('2020-12-31')
WAVELET = 'db4'
WAVELET_MODE = 'zero'
DECOMP_LEVEL = None
THRESHOLD_MODE = 'soft'


def wavelet_denoise_series(x, train_mask, wavelet=WAVELET, level=DECOMP_LEVEL, threshold_mode=THRESHOLD_MODE):
    x = np.ascontiguousarray(x, dtype=np.float64).copy()
    n = len(x)
    if level is None:
        level = pywt.dwt_max_level(n, pywt.Wavelet(wavelet).dec_len)
        level = min(level, 6)
    coeffs = pywt.wavedec(x, wavelet, mode=WAVELET_MODE, level=level)
    cA = coeffs[0]
    cD_list = coeffs[1:]
    finest = cD_list[-1]
    nd = len(finest)
    pos_orig = np.linspace(0, n - 1, nd).astype(int)
    train_in_det = train_mask[pos_orig]
    if train_in_det.sum() < 32:
        train_in_det = np.ones_like(train_in_det, dtype=bool)
    sigma = np.median(np.abs(finest[train_in_det])) / 0.6745
    threshold = sigma * np.sqrt(2.0 * np.log(max(n, 2)))
    cD_thresh = [pywt.threshold(cd, threshold, mode=threshold_mode) for cd in cD_list]
    denoised = pywt.waverec([cA] + cD_thresh, wavelet, mode=WAVELET_MODE)[:n]
    return denoised.astype(np.float64)


def normalise_train_only(data, train_mask):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data[train_mask].reshape(-1, 1))
    scaled = scaler.transform(data.reshape(-1, 1))
    return scaled, scaler


def load_sp500_clean():
    df = pd.read_csv(DATA_DIR / "sp500_clean.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df = df.set_index('Date')
    return df


def main():
    print("=" * 80)
    print("Gerando gráficos xLSTM-TS (modelo treinado)")
    print("=" * 80)

    print("\n[1/4] Carregando dados e modelo...")
    df = load_sp500_clean()
    print(f"   dados: {df.shape}")

    # Carregar modelo treinado
    xlstm_stack, input_projection, output_projection = _patched_create_xlstm_model(SEQ_LENGTH_XLSTM)
    model_path = ROOT / "xlstm_ts_model.pth"
    xlstm_stack.load_state_dict(torch.load(model_path, map_location=DEVICE))
    xlstm_stack.eval()
    print(f"   modelo carregado de {model_path}")

    print("\n[2/4] Preprocessamento (Variante B)...")
    train_mask = np.asarray(df.index <= TRAIN_FIM, dtype=bool)
    close_orig = df["Close"].to_numpy(dtype=float)
    close_denoised = wavelet_denoise_series(close_orig, train_mask)
    close_scaled, scaler = normalise_train_only(close_denoised, train_mask)
    X, y, dates = create_sequences(close_scaled, df.index)

    print("\n[3/4] Split e inferência...")
    STOCK = "S&P 500"
    tr_x, tr_y, tr_d, va_x, va_y, va_d, te_x, te_y, te_d = split_train_val_test_xlstm(
        X, y, dates, TRAIN_END_DATE, VAL_END_DATE, scaler, STOCK
    )

    # Inferência
    with torch.no_grad():
        projected_input_data = input_projection(te_x)
        xlstm_output = xlstm_stack(projected_input_data)
        test_predictions = output_projection(xlstm_output[:, -1, :])
    test_predictions = test_predictions.cpu().numpy().squeeze()
    test_actual = te_y.cpu().numpy().squeeze()
    test_dates = te_d  # dates for x-axis

    print(f"   predições: {len(test_predictions)} dias")

    print("\n[4/4] Gerando gráficos...")
    # Gráfico 1: Previsões vs Real (Teste)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(test_dates, test_actual, label='Real', color='black', linewidth=1.5)
    ax.plot(test_dates, test_predictions, label='Previsão xLSTM-TS', color='red', linewidth=1.5, linestyle='--')
    ax.set_title('xLSTM-TS — Previsões vs Real (Teste 2023-2024)', fontsize=12, weight='bold')
    ax.set_xlabel('Data', fontsize=10)
    ax.set_ylabel('Preço', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'xlstm_ts_predicoes.png', dpi=150)
    plt.close()
    print(f"   salvo: xlstm_ts_predicoes.png")

    # Gráfico 2: Direcional (sobe/desce)
    directions_real = np.diff(test_actual) > 0
    directions_pred = np.diff(test_predictions) > 0
    cm = confusion_matrix(directions_real, directions_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap=plt.cm.Blues, xticklabels=["Desce", "Sobe"], yticklabels=["Desce", "Sobe"], cbar=False, ax=ax)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j + 0.5, i + 0.55, f'\n({cm_norm[i, j]:.2%})',
                     horizontalalignment='center', verticalalignment='center', color='black', fontsize=9)
    ax.set_xlabel('Previsto', fontsize=11, weight='bold')
    ax.set_ylabel('Real', fontsize=11, weight='bold')
    ax.set_title('xLSTM-TS — Matriz de Confusão (Direcional)', fontsize=12, weight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'xlstm_ts_confusion_matrix.png', dpi=150)
    plt.close()
    print(f"   salvo: xlstm_ts_confusion_matrix.png")

    # Gráfico 3: Erro absoluto ao longo do tempo
    errors = np.abs(test_actual - test_predictions)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(te_d, errors, color='orange', linewidth=1)
    ax.set_title('xLSTM-TS — Erro Absoluto ao Longo do Tempo', fontsize=12, weight='bold')
    ax.set_xlabel('Data', fontsize=10)
    ax.set_ylabel('|Erro|', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'xlstm_ts_erro.png', dpi=150)
    plt.close()
    print(f"   salvo: xlstm_ts_erro.png")

    # Gráfico 4: Scatter (Real vs Previsto)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(test_actual, test_predictions, alpha=0.5, s=20)
    min_val = min(test_actual.min(), test_predictions.min())
    max_val = max(test_actual.max(), test_predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')
    ax.set_xlabel('Real', fontsize=11, weight='bold')
    ax.set_ylabel('Previsto', fontsize=11, weight='bold')
    ax.set_title('xLSTM-TS — Scatter Real vs Previsto', fontsize=12, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'xlstm_ts_scatter.png', dpi=150)
    plt.close()
    print(f"   salvo: xlstm_ts_scatter.png")

    print(f"\n→ Gráficos salvos em {PLOTS_DIR}")


if __name__ == "__main__":
    main()
