"""
Transformer-TS — Previsão de preço do S&P 500 com Transformer Encoder (PyTorch).

OBJETIVO: terceiro modelo comparável a LSTM (Variante B) e xLSTM-TS, para o TCC.
Implementado a partir de TRANSFORMER.md, porém ADAPTADO ao projeto:

    - TRANSFORMER.md usa Keras/TensorFlow + yfinance + 5 features (Close, Volume, ...).
      Aqui usamos PyTorch + sp500_clean.csv + 1 feature (Close denoised), EXATAMENTE
      como o xLSTM-TS, para que a comparação seja direta (mesmo pré-processamento,
      mesmo split, mesmo lookback, mesmas métricas).

    - Pré-processamento idêntico ao xLSTM-TS (Variante B):
        * DWT causal (db4, mode='zero', threshold do treino) no Close
        * MinMaxScaler com fit SOMENTE no treino (sem vazamento)
        * Janela deslizante (lookback = SEQ_LENGTH_XLSTM = 150)
    - Split idêntico: 2000-2020 (train) | 2021-2022 (val) | 2023-2024 (test)
    - Tarefa: regressão de preço; direção (sobe/desce) derivada via np.diff
      (igual ao xLSTM-TS), permitindo comparar Acc/F1 direcionais.
    - Métricas calculadas pelas MESMAS funções compartilhadas do repositório
      (calculate_metrics, evaluate_directional_movement) → comparação 1:1.

ARQUITETURA (TRANSFORMER.md, Etapa 4, portada para PyTorch):
    input_projection (Linear 1->d_model) + positional embedding
    -> N blocos TransformerEncoderLayer (Multi-Head Attention + FFN + Add&Norm)
    -> último passo temporal -> cabeça de regressão (Dense->ReLU->Dense(1))

Treinamento alinhado ao xLSTM-TS para comparação justa:
    epochs=200, batch_size=16, lr=1e-4, patience=40, Adam, ReduceLROnPlateau, MSE.

Saída:
    results/transformer_ts_sp500.json
    results/transformer_ts_predicoes.csv
    results/plots/transformer_ts_resultados.png
    transformer_ts_full_checkpoint.pth
"""

import sys
import json
import argparse
import datetime
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None  # noop: evita janelas/plots das funções compartilhadas

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
print(f"[Transformer-TS] DEVICE = {DEVICE}")

# ---------------------------------------------------------------------------
# MONKEY-PATCH: preprocessing._split_data  (.to('cuda') -> .to(DEVICE))
#   Idêntico ao patch usado em xlstm_ts.py para rodar em CPU.
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

# Evita que split_train_val_test_xlstm gere o plot de split (depende de visualisation)
import ml.models.xlstm_ts.preprocessing as _pre_mod
_pre_mod.plot_data_split = lambda *a, **k: None

# ---------------------------------------------------------------------------
# Imports após patches
# ---------------------------------------------------------------------------
from sklearn.preprocessing import MinMaxScaler
import pywt

from ml.constants import SEQ_LENGTH_XLSTM
from ml.models.xlstm_ts.preprocessing import (
    create_sequences,
    split_train_val_test_xlstm,
    inverse_normalise_data_xlstm,
)
from ml.models.shared.metrics import calculate_metrics
from ml.models.shared.directional_prediction import evaluate_directional_movement

# ---------------------------------------------------------------------------
# Denoising causal — Variante B (idêntico a xlstm_ts.py / lstm_base.py)
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

SEED = 42

# Hiperparâmetros do modelo (TRANSFORMER.md, Etapa 3/4) ----------------------
D_MODEL      = 64    # dimensão do embedding (= embedding_dim do xLSTM-TS)
N_HEADS      = 4     # cabeças de atenção (TRANSFORMER.md)
N_LAYERS     = 2     # blocos encoder (TRANSFORMER.md)
DROPOUT_RATE = 0.1
FFN_FACTOR   = 4     # dim_feedforward = D_MODEL * FFN_FACTOR

# Hiperparâmetros de treino (alinhados ao xLSTM-TS p/ comparação justa) ------
LEARNING_RATE = 1e-4
NUM_EPOCHS    = 200
BATCH_SIZE    = 16
PATIENCE      = 40

CHECKPOINT_PATH = ROOT / "transformer_ts_full_checkpoint.pth"

_CAPTURED: dict = {'train_losses': [], 'val_losses': [], 'best_epoch': 0}

MODEL_NAME = "Transformer-TS"
METRIC_KEY = "Transformer-TS_VarianteB"


# ---------------------------------------------------------------------------
# Modelo Transformer (TRANSFORMER.md Etapa 4, portado para PyTorch)
# ---------------------------------------------------------------------------
class TransformerTS(nn.Module):
    """Encoder Transformer para regressão de série temporal univariada.

    Equivalente à build_transformer_model() do TRANSFORMER.md:
        input projection -> positional embedding -> N encoder layers
        -> último passo temporal -> cabeça de regressão.
    """

    def __init__(self, n_features=1, seq_len=SEQ_LENGTH_XLSTM, d_model=D_MODEL,
                 n_heads=N_HEADS, n_layers=N_LAYERS, dropout=DROPOUT_RATE,
                 ffn_factor=FFN_FACTOR):
        super().__init__()
        self.seq_len = seq_len
        self.input_projection = nn.Linear(n_features, d_model)
        self.pos_embedding = nn.Embedding(seq_len, d_model)
        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ffn_factor,
            dropout=dropout,
            activation="relu",
            layer_norm_eps=1e-6,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (B, seq_len, n_features)  — create_sequences entrega (B, seq_len, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        b, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device)
        h = self.input_projection(x) + self.pos_embedding(positions).unsqueeze(0)
        h = self.input_dropout(h)
        h = self.encoder(h)
        h = h[:, -1, :]              # cabeça sobre o último passo temporal ([CLS]-like)
        return self.head(h)          # (B, 1)


# ---------------------------------------------------------------------------
# Treinamento (espelha training.train_model do xLSTM-TS)
# ---------------------------------------------------------------------------
def _make_loader(x, y, batch_size, shuffle):
    ds = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_transformer(model, train_x, train_y, val_x, val_y, num_epochs=NUM_EPOCHS):
    train_loader = _make_loader(train_x, train_y, BATCH_SIZE, shuffle=True)
    val_loader   = _make_loader(val_x, val_y, BATCH_SIZE, shuffle=False)

    criterion = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=10)

    best_val_loss = float('inf')
    best_epoch    = 0
    trigger_times = 0
    initial_lr    = optimiser.param_groups[0]['lr']
    lr_reduced    = False
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        n_batches = 0
        for batch_x, batch_y in train_loader:
            preds = model(batch_x).squeeze()
            loss  = criterion(preds, batch_y.squeeze())
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            epoch_train_loss += loss.item()
            n_batches += 1
        epoch_train_loss /= max(n_batches, 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                preds = model(batch_x).squeeze()
                val_loss += criterion(preds, batch_y.squeeze()).item()
        val_loss /= max(len(val_loader), 1)

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
            torch.save({
                'model':         model.state_dict(),
                'best_epoch':    best_epoch,
                'best_val_loss': best_val_loss,
            }, str(CHECKPOINT_PATH))
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= PATIENCE:
                print('Early stopping!')
                break

    print("Training complete!")
    _CAPTURED['train_losses'] = train_losses
    _CAPTURED['val_losses']   = val_losses
    _CAPTURED['best_epoch']   = best_epoch
    return model


def evaluate_transformer(model, x):
    """Carrega o melhor checkpoint e prevê em x. Retorna tensor (N, 1)."""
    ckpt = torch.load(str(CHECKPOINT_PATH), map_location=DEVICE)
    model.load_state_dict(ckpt['model'])
    model.eval()
    with torch.no_grad():
        preds = model(x)
    return preds


# ---------------------------------------------------------------------------
# Pipeline de execução (espelha run_xlstm_ts da logic.py, sem visualise/darts)
# ---------------------------------------------------------------------------
def run_transformer_ts(train_x, train_y, val_x, val_y, test_x, test_y,
                       scaler, data_type, test_dates, num_epochs=NUM_EPOCHS):
    model = TransformerTS(n_features=1, seq_len=SEQ_LENGTH_XLSTM).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Transformer-TS] parâmetros treináveis: {n_params:,}")

    model = train_transformer(model, train_x, train_y, val_x, val_y, num_epochs)

    # Previsões + inversão da normalização (idêntico ao xLSTM-TS)
    test_predictions  = inverse_normalise_data_xlstm(
        evaluate_transformer(model, test_x).squeeze(), scaler)
    train_predictions = inverse_normalise_data_xlstm(
        evaluate_transformer(model, train_x).squeeze(), scaler)
    val_predictions   = inverse_normalise_data_xlstm(
        evaluate_transformer(model, val_x).squeeze(), scaler)

    test_y_inv  = inverse_normalise_data_xlstm(test_y, scaler)
    train_y_inv = inverse_normalise_data_xlstm(train_y, scaler)
    val_y_inv   = inverse_normalise_data_xlstm(val_y, scaler)

    # Métricas de regressão (mesma função do xLSTM-TS)
    metrics_price = calculate_metrics(test_y_inv, test_predictions, MODEL_NAME, data_type)

    # Métricas direcionais (mesma função do xLSTM-TS, using_darts=False)
    true_labels, predicted_labels, metrics_direction = evaluate_directional_movement(
        train_y_inv, train_predictions, val_y_inv, val_predictions,
        test_y_inv, test_predictions, MODEL_NAME, data_type, using_darts=False,
    )
    metrics_price.update(metrics_direction)

    data = {
        'Date': test_dates.tolist()[:-1],
        'Close': [item for sublist in test_y_inv for item in sublist][:-1],
        'Predicted Value': [item for sublist in test_predictions for item in sublist][:-1],
        'True Label': true_labels.tolist(),
        'Predicted Label': predicted_labels.tolist(),
    }
    results_df = pd.DataFrame(data)
    return results_df, metrics_price


# ---------------------------------------------------------------------------
# Plots (4 painéis, estilo xlstm_ts_resultados.png)
# ---------------------------------------------------------------------------
def generate_plots(results_df: pd.DataFrame, metrics: dict) -> None:
    close     = results_df['Close'].to_numpy()
    predicted = results_df['Predicted Value'].to_numpy()
    dates     = pd.to_datetime(results_df['Date'])
    true_lbl  = results_df['True Label'].to_numpy()
    pred_lbl  = results_df['Predicted Label'].to_numpy()

    correct = (true_lbl == pred_lbl)
    n_total  = len(correct)
    n_correct = int(correct.sum())
    m = metrics.get(METRIC_KEY, {})
    acc_test = m.get("Test Accuracy", 0)
    mape     = m.get("MAPE", 0)
    r2       = m.get("R2", 0)

    train_losses = _CAPTURED['train_losses']
    val_losses   = _CAPTURED['val_losses']
    best_epoch   = _CAPTURED['best_epoch']

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('Resultados — Transformer-TS (Variante B)', fontsize=13, weight='bold')

    # Painel 1: Previsões vs Real
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

    # Painel 2: Loss por Época
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

    # Painel 3: Scatter Real vs Previsto
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

    # Painel 4: Distribuição do Erro Absoluto
    ax = axes[1, 1]
    errors = np.abs(close - predicted)
    mae = m.get("MAE", errors.mean())
    ax.hist(errors, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(mae, color='red', linestyle='--', linewidth=1.5, label=f'MAE={mae:.1f}')
    ax.set_title('Distribuição do Erro Absoluto', fontsize=10)
    ax.set_xlabel('|Erro| (USD)', fontsize=9)
    ax.set_ylabel('Frequência', fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    out_path = PLOTS_DIR / 'transformer_ts_resultados.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"   gráfico salvo: {out_path}")


# ---------------------------------------------------------------------------
# Dados / normalização (idêntico a xlstm_ts.py)
# ---------------------------------------------------------------------------
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


def _clean(d):
    return {k: (float(v) if hasattr(v, "item") else v) for k, v in d.items()}


def _dump(metrics, denoising_info=None, final=False, elapsed_min=None):
    out = {
        "model": MODEL_NAME,
        "variant": "B",
        "preprocessing": "DWT causal (mode=zero, threshold treino) + MinMax (fit treino only)",
        "device": str(DEVICE),
        "ticker": "^GSPC",
        "data_source": "sp500_clean.csv",
        "train_end": TRAIN_END_DATE.isoformat(),
        "val_end": VAL_END_DATE.isoformat(),
        "seq_length": SEQ_LENGTH_XLSTM,
        "model_params": {
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "n_layers": N_LAYERS,
            "dim_feedforward": D_MODEL * FFN_FACTOR,
            "dropout": DROPOUT_RATE,
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LEARNING_RATE,
            "patience": PATIENCE,
            "optimizer": "Adam",
            "scheduler": "ReduceLROnPlateau",
            "loss": "MSE",
        },
        "denoising_info": denoising_info,
        "best_epoch": _CAPTURED.get('best_epoch', None),
        "epochs_run": len(_CAPTURED.get('train_losses', [])),
        "elapsed_min": elapsed_min,
        "metrics": metrics,
        "final": final,
    }
    with open(RESULTS_DIR / "transformer_ts_sp500.json", "w") as f:
        json.dump(out, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Transformer-TS para S&P 500")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help="número de épocas (padrão: 200)")
    parser.add_argument("--quick", action="store_true",
                        help="smoke-test: 3 épocas para validar o pipeline")
    args = parser.parse_args()
    num_epochs = 3 if args.quick else args.epochs

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 80)
    print("Transformer-TS — Variante B (DWT causal mode=zero + MinMax treino only)")
    print("S&P 500 (2000-2024) | Split: 2000-2020 / 2021-2022 / 2023-2024")
    print("=" * 80)

    print("\n[1/6] Carregando dados sp500_clean.csv...")
    df = load_sp500_clean()
    print(f"   shape: {df.shape}  {df.index.min().date()} → {df.index.max().date()}")

    print("\n[2/6] DWT causal denoising (mode='zero', Variante B)...")
    train_mask = np.asarray(df.index <= TRAIN_FIM, dtype=bool)
    close_orig = df["Close"].to_numpy(dtype=float)
    close_denoised, info = wavelet_denoise_series(close_orig, train_mask)
    print(f"   wavelet={info['wavelet']}  level={info['level']}  "
          f"sigma_train={info['sigma_train']:.4f}  threshold={info['threshold']:.4f}  "
          f"pct_zeroed={info['pct_detail_zeroed']:.1f}%  mode={info['wavelet_mode']}")

    print(f"\n[3/6] MinMax (fit=treino only) + criando sequências (lookback={SEQ_LENGTH_XLSTM})...")
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

    print(f"\n[5/6] Treinando Transformer-TS — Variante B  (epochs={num_epochs})")
    print("-" * 80)
    t0 = datetime.datetime.now()
    try:
        results_df, m = run_transformer_ts(
            tr_x, tr_y, va_x, va_y, te_x, te_y, scaler, "VarianteB", te_d, num_epochs,
        )
        metrics[METRIC_KEY] = _clean(m)
    except Exception as e:
        import traceback
        print(f"FAIL: {type(e).__name__}: {e}")
        traceback.print_exc()
        metrics[METRIC_KEY] = {"error": f"{type(e).__name__}: {e}"}
    elapsed = (datetime.datetime.now() - t0).total_seconds() / 60
    print(f"   tempo: {elapsed:.1f} min")

    _dump(metrics, info, final=True, elapsed_min=round(elapsed, 2))
    print(f"\n→ {RESULTS_DIR / 'transformer_ts_sp500.json'}")

    print("\n[6/6] Gerando gráficos e salvando predições...")
    if results_df is not None:
        csv_path = RESULTS_DIR / "transformer_ts_predicoes.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"   predições salvas: {csv_path}")
        generate_plots(results_df, metrics)
    else:
        print("   (gráficos não gerados — erro no treino)")

    print("\nTransformer-TS concluído.")


if __name__ == "__main__":
    main()
