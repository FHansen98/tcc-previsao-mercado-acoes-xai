"""
Utilidades compartilhadas para previsão DIRECIONAL (classificação sobe/desce)
nos modelos xLSTM-TS e Transformer-TS.

Diferenças vs. pipeline de regressão original (xlstm_ts.py / transformer_ts.py):
    - Sequências: y = 1 se o próximo preço sobe, 0 se desce (em vez do valor contínuo)
    - Target derivado do preço ORIGINAL (não denoised) → evita vazamento do denoising
      no rótulo, exatamente como faz a LSTM baseline (lstm_base.criar_targets)
    - Split sem plot/inverse (y é binário)
    - Métricas: Accuracy, F1, MCC, ROC-AUC, Precision, Recall
    - Plots: 4 painéis adaptados para classificação

Mantém-se idêntico ao pipeline de regressão:
    - Input: Close denoised (DWT causal, Variante B) + MinMax (fit treino only)
    - Lookback: SEQ_LENGTH_XLSTM = 150
    - Split temporal: 2000-2020 / 2021-2022 / 2023-2024
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
)

from ml.constants import SEQ_LENGTH_XLSTM


# ---------------------------------------------------------------------------
# Criação de sequências direcionais
# ---------------------------------------------------------------------------
def create_directional_sequences(
    input_scaled: np.ndarray,
    target_close: np.ndarray,
    dates,
    seq_length: int = SEQ_LENGTH_XLSTM,
):
    """Cria sequências para classificação direcional.

    Parameters
    ----------
    input_scaled : np.ndarray (N, 1)
        Close DENOISED + normalizado (MinMax) — alimenta o modelo.
        Mantém o mesmo pré-processamento da regressão (Variante B).
    target_close : np.ndarray (N,)
        Close ORIGINAL (sem denoising) — usado SOMENTE para derivar o rótulo
        de direção, evitando que o denoising contamine o alvo.
    dates : pd.DatetimeIndex | array-like
        Datas alinhadas a input_scaled / target_close.
    seq_length : int
        Tamanho da janela (lookback). Padrão: 150.

    Returns
    -------
    X : torch.FloatTensor (M, seq_length, 1)
    y : torch.FloatTensor (M, 1)  — 0.0 (desce) ou 1.0 (sobe)
    dates_out : pd.Series          — data do ponto previsto (t+1)

    Indexação (alinhada a create_sequences do repo):
        x      = input_scaled[i : i + seq_length]
        alvo   = direção entre target_close[i+seq_length-1] → target_close[i+seq_length]
        data   = dates[i + seq_length]
    """
    input_scaled = np.asarray(input_scaled, dtype=np.float64).reshape(-1, 1)
    target_close = np.asarray(target_close, dtype=np.float64).reshape(-1)
    dates = pd.Series(pd.to_datetime(pd.Series(dates).to_numpy()))

    xs, ys, date_list = [], [], []
    for i in range(len(input_scaled) - seq_length):
        x = input_scaled[i:i + seq_length]
        prev_close = target_close[i + seq_length - 1]
        next_close = target_close[i + seq_length]
        direction = 1.0 if next_close > prev_close else 0.0
        xs.append(x)
        ys.append([direction])
        date_list.append(dates.iloc[i + seq_length])

    X = torch.from_numpy(np.array(xs, dtype=np.float32)).float()
    y = torch.from_numpy(np.array(ys, dtype=np.float32)).float()
    dates_out = pd.Series(date_list)
    return X, y, dates_out


def create_directional_sequences_multifeature(
    input_scaled: np.ndarray,
    target_direction: np.ndarray,
    dates,
    seq_length: int = SEQ_LENGTH_XLSTM,
):
    """Cria sequências para classificação direcional com múltiplas features.

    Parameters
    ----------
    input_scaled : np.ndarray (N, n_features)
        Features normalizadas (Z-score) — alimenta o modelo.
    target_direction : np.ndarray (N,)
        Direção binária (0=desce, 1=sobe) do Close ORIGINAL.
    dates : pd.DatetimeIndex | array-like
        Datas alinhadas a input_scaled / target_direction.
    seq_length : int
        Tamanho da janela (lookback). Padrão: 150.

    Returns
    -------
    X : torch.FloatTensor (M, seq_length, n_features)
    y : torch.FloatTensor (M, 1)  — 0.0 (desce) ou 1.0 (sobe)
    dates_out : pd.Series          — data do ponto previsto (t+1)

    Indexação:
        x      = input_scaled[i : i + seq_length]
        alvo   = target_direction[i + seq_length]
        data   = dates[i + seq_length]
    """
    input_scaled = np.asarray(input_scaled, dtype=np.float64)
    target_direction = np.asarray(target_direction, dtype=np.float64).reshape(-1)
    dates = pd.Series(pd.to_datetime(pd.Series(dates).to_numpy()))

    xs, ys, date_list = [], [], []
    for i in range(len(input_scaled) - seq_length):
        x = input_scaled[i:i + seq_length]
        direction = target_direction[i + seq_length]
        xs.append(x)
        ys.append([direction])
        date_list.append(dates.iloc[i + seq_length])

    X = torch.from_numpy(np.array(xs, dtype=np.float32)).float()
    y = torch.from_numpy(np.array(ys, dtype=np.float32)).float()
    dates_out = pd.Series(date_list)
    return X, y, dates_out


# ---------------------------------------------------------------------------
# Split temporal (sem plot/inverse — y é binário)
# ---------------------------------------------------------------------------
def split_train_val_test_classif(X, y, dates, train_end_date, val_end_date, device):
    """Split temporal idêntico ao do xLSTM-TS, mas sem plot/inverse de y."""
    def _split(name):
        if name == "train":
            mask = (dates < train_end_date)
        elif name == "val":
            mask = (dates >= train_end_date) & (dates < val_end_date)
        elif name == "test":
            mask = (dates >= val_end_date)
        else:
            raise ValueError("set inválido")
        mask = mask.to_numpy()
        x_s = X[mask].to(device)
        y_s = y[mask].to(device)
        print(f"{name} X shape: {x_s.shape}  y shape: {y_s.shape}  "
              f"(% sobe = {float(y_s.mean()) * 100:.1f})")
        return x_s, y_s, dates[mask]

    tr_x, tr_y, tr_d = _split("train")
    va_x, va_y, va_d = _split("val")
    te_x, te_y, te_d = _split("test")
    return tr_x, tr_y, tr_d, va_x, va_y, va_d, te_x, te_y, te_d


# ---------------------------------------------------------------------------
# Métricas de classificação
# ---------------------------------------------------------------------------
def calculate_classification_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    """Calcula métricas de classificação direcional.

    Parameters
    ----------
    y_true  : array-like (N,)  — 0/1 verdadeiros
    y_proba : array-like (N,)  — probabilidade prevista [0,1]
    threshold : float          — corte para binarizar (padrão 0.5)
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_proba = np.asarray(y_proba).astype(float).ravel()
    y_pred = (y_proba > threshold).astype(int)

    out = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc":       float(matthews_corrcoef(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    except Exception:
        out["roc_auc"] = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    out["confusion_matrix"] = cm.tolist()  # [[TN, FP], [FN, TP]]
    out["n_samples"] = int(len(y_true))
    out["pct_up_true"] = float(y_true.mean() * 100)
    out["pct_up_pred"] = float(y_pred.mean() * 100)
    return out


# ---------------------------------------------------------------------------
# Plots (6 painéis para classificação)
# ---------------------------------------------------------------------------
def generate_classification_plots(
    results_df: pd.DataFrame,
    metrics: dict,
    captured: dict,
    model_title: str,
    out_path: Path,
) -> None:
    """Gera figura 6-painéis para classificação direcional.

    results_df precisa conter: Date, True Label, Predicted Label, Predicted Proba
    metrics: dict com accuracy, f1, mcc, roc_auc, confusion_matrix
    captured: dict com train_losses, val_losses, best_epoch
    """
    dates = pd.to_datetime(results_df["Date"])
    true_lbl = results_df["True Label"].to_numpy().astype(int)
    pred_lbl = results_df["Predicted Label"].to_numpy().astype(int)
    proba = results_df["Predicted Proba"].to_numpy().astype(float)

    correct = (true_lbl == pred_lbl)
    n_total = len(correct)
    n_correct = int(correct.sum())

    acc = metrics.get("accuracy", 0) * 100
    f1 = metrics.get("f1", 0)
    mcc = metrics.get("mcc", 0)
    auc = metrics.get("roc_auc", 0)
    precision = metrics.get("precision", 0)
    recall = metrics.get("recall", 0)
    cm = np.array(metrics.get("confusion_matrix", [[0, 0], [0, 0]]))

    train_losses = captured.get("train_losses", [])
    val_losses = captured.get("val_losses", [])
    best_epoch = captured.get("best_epoch", 0)

    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    fig.suptitle(f"Resultados — {model_title} (Classificação Direcional)",
                 fontsize=14, weight="bold")

    # --- Painel 1: Acertos/Erros ao longo do tempo ---
    ax = axes[0, 0]
    ax.set_title(
        f"Previsões no Test Set\nAcc={acc:.1f}%  F1={f1:.3f}  MCC={mcc:.3f}  AUC={auc:.3f}",
        fontsize=10,
    )
    ax.scatter(dates[correct], proba[correct], color="green", s=10, zorder=3,
               label=f"Acerto ({n_correct})", alpha=0.6)
    ax.scatter(dates[~correct], proba[~correct], color="red", s=10, zorder=3,
               label=f"Erro ({n_total - n_correct})", alpha=0.6)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="threshold=0.5")
    ax.set_xlabel("Data", fontsize=9)
    ax.set_ylabel("Probabilidade prevista (sobe)", fontsize=9)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=30, labelsize=8)

    # --- Painel 2: Loss por Época ---
    ax = axes[0, 1]
    if train_losses:
        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, color="steelblue", linewidth=1.5, label="Treino")
        ax.plot(epochs, val_losses, color="orange", linewidth=1.5, label="Val")
        if best_epoch:
            ax.axvline(best_epoch, color="red", linestyle="--", linewidth=1,
                       label=f"best@ep{best_epoch}")
        ax.set_title("Loss por Época (BCE)", fontsize=10)
        ax.set_xlabel("Época", fontsize=9)
        ax.set_ylabel("BCE Loss", fontsize=9)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)
    else:
        ax.text(0.5, 0.5, "Losses não capturadas", ha="center", va="center")

    # --- Painel 3: Matriz de Confusão (aumentada) ---
    ax = axes[1, 0]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Matriz de Confusão", fontsize=10)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Desce", "Sobe"]); ax.set_yticklabels(["Desce", "Sobe"])
    ax.set_xlabel("Previsto", fontsize=9)
    ax.set_ylabel("Real", fontsize=9)
    vmax = cm.max() if cm.max() > 0 else 1
    for r in range(2):
        for c in range(2):
            ax.text(c, r, f"{cm[r, c]}", ha="center", va="center",
                    color="white" if cm[r, c] > vmax / 2 else "black",
                    fontsize=16, weight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # --- Painel 4: Curva ROC ---
    ax = axes[1, 1]
    if auc > 0:
        fpr, tpr, _ = roc_curve(true_lbl, proba)
        ax.plot(fpr, tpr, color="darkorange", linewidth=2, label=f"ROC (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Acaso")
        ax.set_xlabel("Taxa de Falsos Positivos (FPR)", fontsize=9)
        ax.set_ylabel("Taxa de Verdadeiros Positivos (TPR)", fontsize=9)
        ax.set_title("Curva ROC", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
    else:
        ax.text(0.5, 0.5, "AUC não disponível", ha="center", va="center")

    # --- Painel 5: Distribuição das Probabilidades ---
    ax = axes[2, 0]
    ax.hist(proba[true_lbl == 1], bins=30, alpha=0.6, color="green",
            label="Real: sobe", edgecolor="white")
    ax.hist(proba[true_lbl == 0], bins=30, alpha=0.6, color="red",
            label="Real: desce", edgecolor="white")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1.5, label="threshold")
    ax.set_title("Distribuição P(sobe)", fontsize=10)
    ax.set_xlabel("Probabilidade prevista (sobe)", fontsize=9)
    ax.set_ylabel("Frequência", fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

    # --- Painel 6: Métricas Resumidas ---
    ax = axes[2, 1]
    ax.axis("off")
    metrics_text = (
        f"Métricas de Classificação\n\n"
        f"Acurácia:     {acc:.2f}%\n"
        f"F1-Score:     {f1:.4f}\n"
        f"MCC:          {mcc:.4f}\n"
        f"ROC-AUC:      {auc:.4f}\n"
        f"Precisão:     {precision:.4f}\n"
        f"Recall:       {recall:.4f}\n\n"
        f"Amostras:     {n_total}\n"
        f"Acertos:      {n_correct}\n"
        f"Erros:        {n_total - n_correct}"
    )
    ax.text(0.1, 0.5, metrics_text, ha="left", va="center", fontsize=11,
            family="monospace", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"   gráfico salvo: {out_path}")
