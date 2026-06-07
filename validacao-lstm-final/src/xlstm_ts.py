"""
xLSTM-TS com preprocessamento da Variante B (Fase 6, Passo 6).

Variante B: DWT causal (mode='zero', threshold do treino) + MinMax (fit=treino only).
Repositório oficial: https://github.com/gonzalopezgil/xlstm-ts

ALTERAÇÕES vs. paper:
    - sLSTMLayerConfig.backend: "cuda" → "vanilla" (CPU-compatible)
    - .to("cuda") → .to(DEVICE), com DEVICE=auto
    - Dados: usa sp500_clean.csv (2000-2024) em vez de download do yfinance
    - Split: 2000-2020 (train) | 2021-2022 (val) | 2023-2024 (test) — alinhado com LSTM
    - Denoising: wavelet_denoise_series de lstm_base (mode='zero', causal, threshold do treino)
      Substitui wavelet_denoising da biblioteca (mode='per', não causal — vazamento)
    - Scaler: MinMaxScaler fit SOMENTE no treino (sem vazamento)
    - Lookback: 150 dias (padrão do paper)
    - Tarefa: regressão de preço (não classificação direcional como nossa LSTM)
    - Checkpoint completo (xlstm_stack + input_projection + output_projection)
    - Gráficos 4 painéis (estilo resultados_B.png) gerados ao final do treino

NOTA: O backend "vanilla" do sLSTM é numericamente diferente do kernel CUDA original.
Tempo estimado: 4-12h em CPU moderna.

Saída: results/xlstm_ts_sp500.json  |  results/xlstm_ts_predicoes.csv
       results/plots/xlstm_ts_resultados.png
"""

import sys
import json
import datetime
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_SRC = ROOT / "external" / "src"
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(EXTERNAL_SRC))
sys.path.insert(0, str(SRC_DIR))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[xLSTM-TS] DEVICE = {DEVICE}")

# ---------------------------------------------------------------------------
# MONKEY-PATCH 1: xlstm_ts_model.create_xlstm_model
#   - sLSTMLayerConfig.backend: "cuda" -> "vanilla"
#   - .to("cuda") -> .to(DEVICE)
# ---------------------------------------------------------------------------
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
                backend="vanilla",  # CPU-compatible
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

# ---------------------------------------------------------------------------
# MONKEY-PATCH 2: xlstm_ts.preprocessing._split_data
#   .to('cuda') -> .to(DEVICE)
# ---------------------------------------------------------------------------
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
    print(f"{set} X shape: {x_s.shape}  y shape: {y_s.shape}")
    return x_s, y_s, dates[mask]


_xlstm_pre._split_data = _patched_split_data

# ---------------------------------------------------------------------------
# MONKEY-PATCH 3: training.train_model
#   - Captura losses por época em _CAPTURED
#   - Salva checkpoint COMPLETO (xlstm_stack + input_projection + output_projection)
# ---------------------------------------------------------------------------
import ml.models.xlstm_ts.training as _xlstm_training_mod
from ml.models.xlstm_ts.training import create_dataloader

_CAPTURED: dict = {'train_losses': [], 'val_losses': [], 'best_epoch': 0}


def _patched_train_model(xlstm_stack, input_projection, output_projection,
                         train_x, train_y, val_x, val_y):
    learning_rate = 0.0001
    num_epochs    = 200
    batch_size    = 16
    patience      = 40

    train_loader = create_dataloader(train_x, train_y, batch_size, shuffle=True)
    val_loader   = create_dataloader(val_x, val_y, batch_size, shuffle=False)

    criterion  = nn.MSELoss()
    optimiser  = optim.Adam(
        list(xlstm_stack.parameters()) +
        list(input_projection.parameters()) +
        list(output_projection.parameters()),
        lr=learning_rate,
    )
    scheduler  = ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=10)

    best_val_loss  = float('inf')
    best_epoch     = 0
    trigger_times  = 0
    initial_lr     = optimiser.param_groups[0]['lr']
    lr_reduced     = False
    train_losses   = []
    val_losses     = []

    for epoch in range(num_epochs):
        xlstm_stack.train()
        epoch_train_loss = 0.0
        n_batches = 0
        for batch_x, batch_y in train_loader:
            proj   = input_projection(batch_x)
            out    = xlstm_stack(proj)
            preds  = output_projection(out[:, -1, :]).squeeze()
            loss   = criterion(preds, batch_y.squeeze())
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(xlstm_stack.parameters(), max_norm=1.0)
            optimiser.step()
            epoch_train_loss += loss.item()
            n_batches += 1
        epoch_train_loss /= max(n_batches, 1)

        xlstm_stack.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                proj  = input_projection(batch_x)
                out   = xlstm_stack(proj)
                preds = output_projection(out[:, -1, :]).squeeze()
                val_loss += criterion(preds, batch_y.squeeze()).item()
        val_loss /= len(val_loader)

        train_losses.append(epoch_train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        if not lr_reduced and optimiser.param_groups[0]['lr'] < initial_lr:
            print(f'Epoch [{epoch+1}/{num_epochs}], Reducing LR to '
                  f'{optimiser.param_groups[0]["lr"]}')
            lr_reduced = True

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Loss: {epoch_train_loss:.8f}, Validation Loss: {val_loss:.8f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch + 1
            torch.save(xlstm_stack.state_dict(), 'xlstm_ts_model.pth')
            # Checkpoint COMPLETO: salva os 3 componentes
            torch.save({
                'xlstm_stack':        xlstm_stack.state_dict(),
                'input_projection':   input_projection.state_dict(),
                'output_projection':  output_projection.state_dict(),
                'best_epoch':         best_epoch,
                'best_val_loss':      best_val_loss,
            }, str(ROOT / 'xlstm_ts_full_checkpoint.pth'))
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print('Early stopping!')
                break

    print("Training complete!")
    _CAPTURED['train_losses'] = train_losses
    _CAPTURED['val_losses']   = val_losses
    _CAPTURED['best_epoch']   = best_epoch
    return xlstm_stack, input_projection, output_projection


_xlstm_training_mod.train_model = _patched_train_model

# Patch no namespace de logic.py (onde run_xlstm_ts resolve o nome)
import ml.models.xlstm_ts.logic as _xlstm_logic_mod
_xlstm_logic_mod.train_model = _patched_train_model

# ---------------------------------------------------------------------------
# Imports após todos os patches
# ---------------------------------------------------------------------------
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import pywt

from ml.models.xlstm_ts.preprocessing import (
    create_sequences,
    split_train_val_test_xlstm,
)
from ml.models.xlstm_ts.logic import run_xlstm_ts

# ---------------------------------------------------------------------------
# Denoising causal — Variante B (extraído de lstm_base sem dep. TensorFlow)
# ---------------------------------------------------------------------------
TRAIN_FIM      = pd.Timestamp('2020-12-31')
WAVELET        = 'db4'
WAVELET_MODE   = 'zero'
DECOMP_LEVEL   = None
THRESHOLD_MODE = 'soft'


def wavelet_denoise_series(
    x: np.ndarray,
    train_mask: np.ndarray,
    wavelet: str = WAVELET,
    level: int | None = DECOMP_LEVEL,
    threshold_mode: str = THRESHOLD_MODE,
) -> tuple:
    """DWT denoising causal (mode='zero'). Idêntico a lstm_base.wavelet_denoise_series."""
    x = np.ascontiguousarray(x, dtype=np.float64).copy()
    n = len(x)
    if level is None:
        level = pywt.dwt_max_level(n, pywt.Wavelet(wavelet).dec_len)
        level = min(level, 6)
    coeffs   = pywt.wavedec(x, wavelet, mode=WAVELET_MODE, level=level)
    cA       = coeffs[0]
    cD_list  = coeffs[1:]
    finest   = cD_list[-1]
    nd           = len(finest)
    pos_orig     = np.linspace(0, n - 1, nd).astype(int)
    train_in_det = train_mask[pos_orig]
    if train_in_det.sum() < 32:
        train_in_det = np.ones_like(train_in_det, dtype=bool)
    sigma     = np.median(np.abs(finest[train_in_det])) / 0.6745
    threshold = sigma * np.sqrt(2.0 * np.log(max(n, 2)))
    cD_thresh = [pywt.threshold(cd, threshold, mode=threshold_mode) for cd in cD_list]
    denoised  = pywt.waverec([cA] + cD_thresh, wavelet, mode=WAVELET_MODE)[:n]
    total_det = sum(len(cd) for cd in cD_list)
    zeroed    = sum(int((np.abs(cd) <= threshold).sum()) for cd in cD_list)
    info = {
        'wavelet': wavelet, 'level': int(level),
        'sigma_train': float(sigma), 'threshold': float(threshold),
        'pct_detail_zeroed': zeroed / max(total_det, 1) * 100.0,
        'wavelet_mode': WAVELET_MODE,
    }
    return denoised.astype(np.float64), info


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

TRAIN_END_DATE = datetime.datetime.strptime("2020-12-31", "%Y-%m-%d")
VAL_END_DATE   = datetime.datetime.strptime("2022-12-31", "%Y-%m-%d")


def load_sp500_clean() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "sp500_clean.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df = df.set_index('Date')
    return df


def normalise_train_only(data: np.ndarray, train_mask: np.ndarray):
    """MinMaxScaler fit SOMENTE no treino — sem vazamento de dados."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data[train_mask].reshape(-1, 1))
    scaled = scaler.transform(data.reshape(-1, 1))
    return scaled, scaler


def generate_plots(results_df: pd.DataFrame, metrics: dict) -> None:
    """Gera figura 4-painéis estilo resultados_B.png."""
    close     = results_df['Close'].to_numpy()
    predicted = results_df['Predicted Value'].to_numpy()
    dates     = pd.to_datetime(results_df['Date'])
    true_lbl  = results_df['True Label'].to_numpy()
    pred_lbl  = results_df['Predicted Label'].to_numpy()

    correct = (true_lbl == pred_lbl)
    n_total  = len(correct)
    n_correct = correct.sum()
    acc_test = metrics.get("xLSTM-TS_VarianteB", {}).get("Test Accuracy", 0)
    mape     = metrics.get("xLSTM-TS_VarianteB", {}).get("MAPE", 0)
    r2       = metrics.get("xLSTM-TS_VarianteB", {}).get("R2", 0)

    train_losses = _CAPTURED['train_losses']
    val_losses   = _CAPTURED['val_losses']
    best_epoch   = _CAPTURED['best_epoch']

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('Resultados — xLSTM-TS (Variante B)', fontsize=13, weight='bold')

    # --- Painel 1: Previsões vs Real ---
    ax = axes[0, 0]
    ax.set_title(f'Previsões no Test Set\nAcc={acc_test:.1f}%  MAPE={mape:.2f}%  R²={r2:.3f}',
                 fontsize=10)
    ax.plot(dates, close, color='navy', linewidth=1.2, label='Close real', zorder=2)
    ax.scatter(dates[correct],  close[correct],  color='green', s=10, zorder=3,
               label=f'Acerto ({n_correct})')
    ax.scatter(dates[~correct], close[~correct], color='red',   s=10, zorder=3,
               label=f'Erro ({n_total - n_correct})')
    ax.plot(dates, predicted, color='gray', linewidth=0.9, linestyle='--',
            label='Previsto', zorder=2, alpha=0.7)
    ax.set_xlabel('Data', fontsize=9)
    ax.set_ylabel('USD', fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(axis='x', rotation=30, labelsize=8)

    # --- Painel 2: Loss por Época ---
    ax = axes[0, 1]
    if train_losses:
        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, color='steelblue', linewidth=1.5, label='Treino')
        ax.plot(epochs, val_losses,   color='orange',    linewidth=1.5, label='Val')
        if best_epoch:
            ax.axvline(best_epoch, color='red', linestyle='--', linewidth=1,
                       label=f'best@ep{best_epoch}')
        ax.set_title('Loss por Época', fontsize=10)
        ax.set_xlabel('Época', fontsize=9)
        ax.set_ylabel('MSE Loss', fontsize=9)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)
    else:
        ax.text(0.5, 0.5, 'Losses não capturadas', ha='center', va='center')

    # --- Painel 3: Scatter Real vs Previsto ---
    ax = axes[1, 0]
    ax.scatter(close, predicted, alpha=0.4, s=12, color='steelblue')
    mn = min(close.min(), predicted.min())
    mx = max(close.max(), predicted.max())
    ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5, label='y = x')
    ax.set_title(f'Scatter Real vs Previsto  (R²={r2:.3f})', fontsize=10)
    ax.set_xlabel('Real (USD)', fontsize=9)
    ax.set_ylabel('Previsto (USD)', fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

    # --- Painel 4: Distribuição do Erro Absoluto ---
    ax = axes[1, 1]
    errors = np.abs(close - predicted)
    mae = metrics.get("xLSTM-TS_VarianteB", {}).get("MAE", errors.mean())
    ax.hist(errors, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(mae, color='red', linestyle='--', linewidth=1.5, label=f'MAE={mae:.1f}')
    ax.set_title('Distribuição do Erro Absoluto', fontsize=10)
    ax.set_xlabel('|Erro| (USD)', fontsize=9)
    ax.set_ylabel('Frequência', fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    out_path = PLOTS_DIR / 'xlstm_ts_resultados.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"   gráfico salvo: {out_path}")


def main():
    print("=" * 80)
    print("xLSTM-TS — Variante B (DWT causal mode=zero + MinMax treino only)")
    print("S&P 500 (2000-2024) | Split: 2000-2020 / 2021-2022 / 2023-2024")
    print("=" * 80)

    print(f"\n[1/6] Carregando dados sp500_clean.csv...")
    df = load_sp500_clean()
    print(f"   shape: {df.shape}  {df.index.min().date()} → {df.index.max().date()}")

    print("\n[2/6] DWT causal denoising (mode='zero', Variante B)...")
    train_mask = np.asarray(df.index <= TRAIN_FIM, dtype=bool)
    close_orig = df["Close"].to_numpy(dtype=float)
    close_denoised, info = wavelet_denoise_series(close_orig, train_mask)
    print(f"   wavelet={info['wavelet']}  level={info['level']}  "
          f"sigma_train={info['sigma_train']:.4f}  threshold={info['threshold']:.4f}  "
          f"pct_zeroed={info['pct_detail_zeroed']:.1f}%  mode={info['wavelet_mode']}")

    print("\n[3/6] MinMax (fit=treino only) + criando sequências (lookback=150)...")
    close_scaled, scaler = normalise_train_only(close_denoised, train_mask)
    X, y, dates = create_sequences(close_scaled, df.index)
    print(f"   sequências: X={X.shape}  y={y.shape}")

    print("\n[4/6] Split train/val/test...")
    STOCK = "S&P 500"
    tr_x, tr_y, tr_d, va_x, va_y, va_d, te_x, te_y, te_d = split_train_val_test_xlstm(
        X, y, dates, TRAIN_END_DATE, VAL_END_DATE, scaler, STOCK
    )

    metrics = {}
    results_df = None

    print("\n[5/6] Treinando xLSTM-TS — Variante B")
    print("-" * 80)
    t0 = datetime.datetime.now()
    try:
        results_df, m = run_xlstm_ts(
            tr_x, tr_y, va_x, va_y, te_x, te_y,
            scaler, STOCK, "VarianteB", te_d,
        )
        metrics["xLSTM-TS_VarianteB"] = _clean(m)
    except Exception as e:
        import traceback
        print(f"FAIL: {type(e).__name__}: {e}")
        traceback.print_exc()
        metrics["xLSTM-TS_VarianteB"] = {"error": f"{type(e).__name__}: {e}"}
    elapsed = (datetime.datetime.now() - t0).total_seconds() / 60
    print(f"   tempo: {elapsed:.1f} min")

    _dump(metrics, info, final=True)
    print(f"\n→ {RESULTS_DIR / 'xlstm_ts_sp500.json'}")

    print("\n[6/6] Gerando gráficos e salvando predições...")
    if results_df is not None:
        csv_path = RESULTS_DIR / "xlstm_ts_predicoes.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"   predições salvas: {csv_path}")
        generate_plots(results_df, metrics)
    else:
        print("   (gráficos não gerados — erro no treino)")

    print("\nxLSTM-TS concluído.")


def _clean(d):
    return {k: (float(v) if hasattr(v, "item") else v) for k, v in d.items()}


def _dump(metrics, denoising_info=None, final=False):
    out = {
        "variant": "B",
        "preprocessing": "DWT causal (mode=zero, threshold treino) + MinMax (fit treino only)",
        "device": str(DEVICE),
        "backend_slstm": "vanilla",
        "ticker": "^GSPC",
        "data_source": "sp500_clean.csv",
        "train_end": TRAIN_END_DATE.isoformat(),
        "val_end": VAL_END_DATE.isoformat(),
        "seq_length_xlstm": SEQ_LENGTH_XLSTM,
        "model_params": {
            "num_blocks": 4,
            "embedding_dim": 64,
            "num_heads": 2,
            "num_epochs": 200,
            "batch_size": 16,
            "lr": 1e-4,
            "patience": 40,
            "scheduler": "ReduceLROnPlateau",
        },
        "denoising_info": denoising_info,
        "best_epoch": _CAPTURED.get('best_epoch', None),
        "epochs_run": len(_CAPTURED.get('train_losses', [])),
        "metrics": metrics,
        "final": final,
    }
    with open(RESULTS_DIR / "xlstm_ts_sp500.json", "w") as f:
        json.dump(out, f, indent=2, default=str)


if __name__ == "__main__":
    main()
