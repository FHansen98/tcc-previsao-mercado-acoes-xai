"""
xAI Analysis — SHAP para xLSTM-TS e Transformer-TS.

Objetivo:
  - Aplicar SHAP DeepExplainer nos modelos treinados (xLSTM-TS e Transformer-TS)
  - Interpretar importância temporal (quais timesteps mais influenciam previsões)
  - Gerar visualizações e métricas de interpretabilidade

Modelos:
  - xLSTM-TS: regressor (1 feature, lookback=150)
  - Transformer-TS: regressor (1 feature, lookback=150)

IMPORTANTE — Prevenção de vazamento de dados:
  - Background dataset deve vir APENAS do treino (nunca do teste)
  - SHAP values calculados para amostras do teste
  - Seed fixo para reprodutibilidade

Dependências necessárias:
  - torch>=2.0.0
  - xlstm>=1.0.0 (apenas para xLSTM-TS)
  - shap>=0.45.0
  - numpy, pandas, matplotlib, pywt, scikit-learn

Execução:
  python src/xai_analysis.py --model xlstm
  python src/xai_analysis.py --model transformer
  python src/xai_analysis.py --model both
"""

from __future__ import annotations

import json
import datetime
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import pywt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# PyTorch e SHAP (só importados se disponíveis)
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("AVISO: torch não disponível. Script não pode rodar sem torch.")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("AVISO: shap não disponível. Script não pode rodar sem shap.")

# xlstm (só importado se disponível e necessário)
try:
    from xlstm import (
        xLSTMBlockStack, xLSTMBlockStackConfig,
        mLSTMBlockConfig, mLSTMLayerConfig,
        sLSTMBlockConfig, sLSTMLayerConfig,
        FeedForwardConfig,
    )
    XLSTM_AVAILABLE = True
except ImportError:
    XLSTM_AVAILABLE = False
    print("AVISO: xlstm não disponível. xLSTM-TS não pode ser carregado.")

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
XAI_DIR = RESULTS_DIR / "xai"
XAI_PLOTS_DIR = XAI_DIR / "plots"

XAI_DIR.mkdir(exist_ok=True, parents=True)
XAI_PLOTS_DIR.mkdir(exist_ok=True, parents=True)

TRAIN_END_DATE = datetime.datetime.strptime("2020-12-31", "%Y-%m-%d")
VAL_END_DATE = datetime.datetime.strptime("2022-12-31", "%Y-%m-%d")

SEQ_LENGTH = 150  # Lookback (igual para ambos os modelos)
SEED = 42
BACKGROUND_SAMPLES = 100  # Número de amostras do treino para background
TEST_SAMPLES = 200  # Número de amostras do teste para SHAP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Wavelet — Variante B (idêntico a xlstm_ts.py e transformer_ts.py)
WAVELET = 'db4'
WAVELET_MODE = 'zero'
DECOMP_LEVEL = None
THRESHOLD_MODE = 'soft'

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(ROOT / 'logs' / 'xai_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Funções de pré-processamento (reutilizadas de xlstm_ts.py/transformer_ts.py)
# ---------------------------------------------------------------------------
def load_sp500_clean() -> pd.DataFrame:
    """Carrega dados limpos do S&P 500."""
    df = pd.read_csv(DATA_DIR / "sp500_clean.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df = df.set_index('Date')
    return df


def wavelet_denoise_series(
    x: np.ndarray,
    train_mask: np.ndarray,
    wavelet: str = WAVELET,
    level: int | None = DECOMP_LEVEL,
    threshold_mode: str = THRESHOLD_MODE,
) -> tuple:
    """DWT denoising causal (mode='zero'). Idêntico a xlstm_ts.py/transformer_ts.py."""
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
    info = {
        'wavelet': wavelet, 'level': int(level),
        'sigma_train': float(sigma), 'threshold': float(threshold),
        'pct_detail_zeroed': sum(int((np.abs(cd) <= threshold).sum()) for cd in cD_list) / 
                             max(sum(len(cd) for cd in cD_list), 1) * 100.0,
        'wavelet_mode': WAVELET_MODE,
    }
    return denoised.astype(np.float64), info


def normalise_train_only(data: np.ndarray, train_mask: np.ndarray):
    """MinMaxScaler fit SOMENTE no treino — sem vazamento de dados."""
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data[train_mask].reshape(-1, 1))
    scaled = scaler.transform(data.reshape(-1, 1))
    return scaled, scaler


def create_sequences(data, dates):
    """Cria sequências com lookback=SEQ_LENGTH."""
    xs, ys, date_list = [], [], []
    for i in range(len(data) - SEQ_LENGTH):
        x = data[i:i + SEQ_LENGTH]
        y = data[i + SEQ_LENGTH]
        date = dates[i + SEQ_LENGTH]
        xs.append(x)
        ys.append(y)
        date_list.append(date)
    X = np.array(xs)
    y = np.array(ys)
    dates = pd.Series(date_list)
    return X, y, dates


def split_train_val_test(X, y, dates):
    """Split temporal train/val/test."""
    train_mask = dates < TRAIN_END_DATE
    val_mask = (dates >= TRAIN_END_DATE) & (dates < VAL_END_DATE)
    test_mask = dates >= VAL_END_DATE
    
    train_x, train_y, train_d = X[train_mask], y[train_mask], dates[train_mask]
    val_x, val_y, val_d = X[val_mask], y[val_mask], dates[val_mask]
    test_x, test_y, test_d = X[test_mask], y[test_mask], dates[test_mask]
    
    return train_x, train_y, train_d, val_x, val_y, val_d, test_x, test_y, test_d


# ---------------------------------------------------------------------------
# Wrappers para SHAP
# ---------------------------------------------------------------------------
class xLSTMWrapper(nn.Module):
    """Wrapper para SHAP: xLSTM-TS com 3 componentes."""
    
    def __init__(self, xlstm_stack, input_projection, output_projection):
        super().__init__()
        self.xlstm_stack = xlstm_stack
        self.input_projection = input_projection
        self.output_projection = output_projection
    
    def forward(self, x):
        # x: [batch, seq_len, 1]
        x_proj = self.input_projection(x)  # [batch, seq_len, 64]
        xlstm_out = self.xlstm_stack(x_proj)  # [batch, seq_len, 64]
        # Pegar último timestep
        last_timestep = xlstm_out[:, -1, :]  # [batch, 64]
        return self.output_projection(last_timestep)  # [batch, 1]


class TransformerWrapper(nn.Module):
    """Wrapper para SHAP: Transformer-TS (já é um módulo PyTorch)."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        return self.model(x)


# ---------------------------------------------------------------------------
# Carregamento de Modelos
# ---------------------------------------------------------------------------
def load_xlstm_model(checkpoint_path: Path) -> Tuple[nn.Module, Dict]:
    """Carrega checkpoint do xLSTM-TS e reconstrói o modelo."""
    if not XLSTM_AVAILABLE:
        raise ImportError("xlstm não disponível. Não é possível carregar xLSTM-TS.")
    
    logger.info(f"Carregando checkpoint xLSTM-TS: {checkpoint_path}")
    
    # Configuração idêntica a xlstm_ts.py
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
        context_length=SEQ_LENGTH,
        num_blocks=4,
        embedding_dim=64,
        slstm_at=[1],
    )
    
    xlstm_stack = xLSTMBlockStack(cfg).to(DEVICE)
    input_projection = nn.Linear(1, 64).to(DEVICE)
    output_projection = nn.Linear(64, 1).to(DEVICE)
    
    # Carregar checkpoint
    ckpt = torch.load(str(checkpoint_path), map_location=DEVICE, weights_only=False)
    xlstm_stack.load_state_dict(ckpt['xlstm_stack'])
    input_projection.load_state_dict(ckpt['input_projection'])
    output_projection.load_state_dict(ckpt['output_projection'])
    
    # Criar wrapper
    model = xLSTMWrapper(xlstm_stack, input_projection, output_projection).to(DEVICE)
    model.eval()
    
    info = {
        'best_epoch': ckpt.get('best_epoch', None),
        'best_val_loss': ckpt.get('best_val_loss', None),
    }
    
    logger.info(f"xLSTM-TS carregado: best_epoch={info['best_epoch']}, best_val_loss={info['best_val_loss']}")
    return model, info


def load_transformer_model(checkpoint_path: Path) -> Tuple[nn.Module, Dict]:
    """Carrega checkpoint do Transformer-TS e reconstrói o modelo."""
    logger.info(f"Carregando checkpoint Transformer-TS: {checkpoint_path}")
    
    # Arquitetura idêntica a transformer_ts.py
    class TransformerTS(nn.Module):
        def __init__(self, n_features=1, seq_len=SEQ_LENGTH, d_model=64,
                     n_heads=4, n_layers=2, dropout=0.1, ffn_factor=4):
            super().__init__()
            self.seq_len = seq_len
            self.input_projection = nn.Linear(n_features, d_model)
            self.pos_embedding = nn.Embedding(seq_len, d_model)
            self.input_dropout = nn.Dropout(dropout)
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_model * ffn_factor, dropout=dropout,
                activation="relu", layer_norm_eps=1e-6,
                batch_first=True, norm_first=False,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            
            self.head = nn.Sequential(
                nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1),
            )
        
        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(-1)
            b, seq_len, _ = x.shape
            positions = torch.arange(seq_len, device=x.device)
            h = self.input_projection(x) + self.pos_embedding(positions).unsqueeze(0)
            h = self.input_dropout(h)
            h = self.encoder(h)
            h = h[:, -1, :]
            return self.head(h)
    
    model = TransformerTS().to(DEVICE)
    
    # Carregar checkpoint
    ckpt = torch.load(str(checkpoint_path), map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    # Criar wrapper
    wrapper = TransformerWrapper(model).to(DEVICE)
    
    info = {
        'best_epoch': ckpt.get('best_epoch', None),
        'best_val_loss': ckpt.get('best_val_loss', None),
    }
    
    logger.info(f"Transformer-TS carregado: best_epoch={info['best_epoch']}, best_val_loss={info['best_val_loss']}")
    return wrapper, info


# ---------------------------------------------------------------------------
# Preparação de Background e Test Data
# ---------------------------------------------------------------------------
def prepare_background_and_test(
    train_x: np.ndarray,
    test_x: np.ndarray,
    n_background: int = BACKGROUND_SAMPLES,
    n_test: int = TEST_SAMPLES,
    seed: int = SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepara background dataset (do treino) e test subset.
    
    IMPORTANTE: Background vem APENAS do treino para evitar vazamento.
    """
    np.random.seed(seed)
    
    # Background: amostras aleatórias do treino
    background_indices = np.random.choice(len(train_x), min(n_background, len(train_x)), replace=False)
    background = train_x[background_indices]
    
    # Test subset: primeiras n_test amostras do teste
    test_subset = test_x[:min(n_test, len(test_x))]
    
    logger.info(f"Background shape: {background.shape} (do treino)")
    logger.info(f"Test subset shape: {test_subset.shape} (do teste)")
    
    return background, test_subset, background_indices


# ---------------------------------------------------------------------------
# Cálculo de SHAP Values
# ---------------------------------------------------------------------------
def compute_shap_values(
    model: nn.Module,
    background: np.ndarray,
    test_subset: np.ndarray,
    model_name: str,
) -> Tuple[np.ndarray, float]:
    """
    Calcula SHAP values usando DeepExplainer ou GradientExplainer.
    
    GradientExplainer é usado como fallback para modelos complexos (ex: Transformer com LayerNorm).
    
    Args:
        model: Modelo PyTorch (wrapper)
        background: Background dataset (do treino)
        test_subset: Amostras do teste para explicar
        model_name: Nome do modelo (para logging)
    
    Returns:
        shap_values: Array de SHAP values [n_test, seq_len, 1]
        expected_value: Valor esperado (baseline)
    """
    if not SHAP_AVAILABLE:
        raise ImportError("shap não disponível. Não é possível calcular SHAP values.")
    
    logger.info(f"Calculando SHAP values para {model_name}...")
    
    # Converter para torch tensors
    background_tensor = torch.from_numpy(background).float().to(DEVICE)
    test_tensor = torch.from_numpy(test_subset).float().to(DEVICE)
    
    # Tentar DeepExplainer primeiro (mais rápido)
    try:
        logger.info("Tentando DeepExplainer...")
        explainer = shap.DeepExplainer(model, background_tensor)
        shap_values = explainer.shap_values(test_tensor, check_additivity=False)
        expected_value = explainer.expected_value
        logger.info(f"DeepExplainer funcionou. SHAP values shape: {shap_values.shape}")
    except (AssertionError, RuntimeError) as e:
        logger.warning(f"DeepExplainer falhou: {e}")
        logger.info("Usando PermutationExplainer como fallback (mais lento, mas robusto)...")
        
        # PermutationExplainer (mais genérico, funciona com qualquer modelo)
        def model_wrapper(x):
            x_tensor = torch.from_numpy(x).float().to(DEVICE)
            return model(x_tensor).detach().cpu().numpy()
        
        explainer = shap.PermutationExplainer(model_wrapper, shap.sample(background, 50))
        shap_obj = explainer(test_subset)
        shap_values = shap_obj.values
        expected_value = shap_obj.expected_value
        logger.info(f"PermutationExplainer funcionou. SHAP values shape: {shap_values.shape}")
    
    logger.info(f"Expected value: {expected_value}")
    
    return shap_values, expected_value


# ---------------------------------------------------------------------------
# Análise de SHAP Values
# ---------------------------------------------------------------------------
def analyze_temporal_importance(shap_values: np.ndarray) -> Dict[str, Any]:
    """
    Analisa importância temporal dos timesteps.
    
    Args:
        shap_values: [n_samples, seq_len, 1] ou [n_samples, seq_len, 1, 1]
    
    Returns:
        Dict com métricas de importância temporal
    """
    # Remover dimensões extras se necessário
    if shap_values.ndim == 4:
        shap_values = shap_values.squeeze(-1)  # [n_samples, seq_len, 1]
    if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
        shap_values = shap_values.squeeze(-1)  # [n_samples, seq_len]
    
    # Importância média por timestep (mean absolute SHAP)
    temporal_importance = np.mean(np.abs(shap_values), axis=0)  # [seq_len]
    
    # Ranking de timesteps
    ranked_timesteps = np.argsort(temporal_importance)[::-1]  # Decrescente
    
    # Top timesteps
    top_timesteps = ranked_timesteps[:10].tolist()
    top_importance = temporal_importance[ranked_timesteps[:10]].tolist()
    
    # Cumulative importance
    sorted_importance = temporal_importance[ranked_timesteps]
    cumulative = np.cumsum(sorted_importance) / sorted_importance.sum()
    
    # Quantos timesteps explicam 90% da importância
    n_timesteps_90pct = np.argmax(cumulative >= 0.9) + 1
    
    return {
        'temporal_importance': temporal_importance.tolist(),
        'top_timesteps': top_timesteps,
        'top_importance': top_importance,
        'n_timesteps_90pct': int(n_timesteps_90pct),
        'cumulative_importance': cumulative.tolist(),
    }


# ---------------------------------------------------------------------------
# Visualizações
# ---------------------------------------------------------------------------
def generate_shap_plots(
    shap_values: np.ndarray,
    temporal_importance: np.ndarray,
    model_name: str,
    output_dir: Path,
):
    """Gera visualizações de SHAP."""
    logger.info(f"Gerando plots para {model_name}...")
    
    # Remover dimensões extras se necessário
    if shap_values.ndim == 4:
        shap_values = shap_values.squeeze(-1)  # [n_samples, seq_len, 1]
    if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
        shap_values = shap_values.squeeze(-1)  # [n_samples, seq_len]
    
    # Garantir que temporal_importance seja 1D
    temporal_importance = np.asarray(temporal_importance).flatten()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'SHAP Analysis — {model_name}', fontsize=14, weight='bold')
    
    # --- Painel 1: Importância temporal (line plot) ---
    ax = axes[0, 0]
    timesteps = np.arange(1, len(temporal_importance) + 1)
    ax.plot(timesteps, temporal_importance, color='steelblue', linewidth=2)
    ax.set_xlabel('Timestep (dias passados)', fontsize=10)
    ax.set_ylabel('Importância média (|SHAP|)', fontsize=10)
    ax.set_title('Importância Temporal por Timestep', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)
    
    # --- Painel 2: Heatmap de magnitude |SHAP| por timestep e amostra ---
    ax = axes[0, 1]
    # Magnitude |SHAP| (mesma métrica do gráfico de barras, garantindo coerência entre painéis)
    shap_mag = np.abs(shap_values)
    vmax = np.percentile(shap_mag, 99)
    im = ax.imshow(shap_mag.T, aspect='auto', cmap='viridis', vmin=0, vmax=vmax)
    ax.set_xlabel('Amostra do teste', fontsize=10)
    ax.set_ylabel('Timestep', fontsize=10)
    ax.set_title('Heatmap |SHAP| por timestep', fontsize=11)
    plt.colorbar(im, ax=ax, label='|SHAP|')
    ax.tick_params(labelsize=9)
    
    # --- Painel 3: Top timesteps ---
    ax = axes[1, 0]
    top_n = 10
    ranked = np.argsort(temporal_importance)[::-1][:top_n]
    ax.barh(range(top_n), temporal_importance[ranked], color='steelblue')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([f't-{i}' for i in ranked + 1])
    ax.set_xlabel('Importância média (|SHAP|)', fontsize=10)
    ax.set_title(f'Top {top_n} Timesteps Mais Importantes', fontsize=11)
    ax.invert_yaxis()
    ax.tick_params(labelsize=9)
    
    # --- Painel 4: Cumulative importance ---
    ax = axes[1, 1]
    sorted_importance = np.sort(temporal_importance)[::-1]
    cumulative = np.cumsum(sorted_importance) / sorted_importance.sum()
    timesteps_cum = np.arange(1, len(cumulative) + 1)
    ax.plot(timesteps_cum, cumulative, color='darkgreen', linewidth=2)
    ax.axhline(0.9, color='red', linestyle='--', linewidth=1, label='90%')
    ax.set_xlabel('Número de timesteps (acumulados)', fontsize=10)
    ax.set_ylabel('Importância acumulada', fontsize=10)
    ax.set_title('Importância Acumulada por Timestep', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)
    
    plt.tight_layout()
    output_path = output_dir / f'shap_{model_name.lower().replace("-", "_")}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Plot salvo: {output_path}")


# ---------------------------------------------------------------------------
# Pipeline Principal
# ---------------------------------------------------------------------------
def run_xai_for_model(model_name: str):
    """Executa análise xAI para um modelo específico."""
    logger.info("=" * 80)
    logger.info(f"xAI Analysis — {model_name}")
    logger.info("=" * 80)
    
    # Verificar dependências
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch não disponível. Instale torch para rodar xAI.")
    if not SHAP_AVAILABLE:
        raise RuntimeError("shap não disponível. Instale shap para rodar xAI.")
    if model_name == "xlstm" and not XLSTM_AVAILABLE:
        raise RuntimeError("xlstm não disponível. Instale xlstm para rodar xAI no xLSTM-TS.")
    
    # 1. Carregar dados
    logger.info("[1/7] Carregando dados...")
    df = load_sp500_clean()
    logger.info(f"   shape: {df.shape}  {df.index.min().date()} → {df.index.max().date()}")
    
    # 2. Denoising (Variante B)
    logger.info("[2/7] DWT causal denoising (Variante B)...")
    train_mask = np.asarray(df.index <= TRAIN_END_DATE, dtype=bool)
    close_orig = df["Close"].to_numpy(dtype=float)
    close_denoised, denoising_info = wavelet_denoise_series(close_orig, train_mask)
    logger.info(f"   wavelet={denoising_info['wavelet']}  level={denoising_info['level']}  "
                f"sigma_train={denoising_info['sigma_train']:.4f}  "
                f"threshold={denoising_info['threshold']:.4f}  "
                f"pct_zeroed={denoising_info['pct_detail_zeroed']:.1f}%")
    
    # 3. Normalização + sequências
    logger.info("[3/7] MinMax (fit=treino only) + sequências...")
    close_scaled, scaler = normalise_train_only(close_denoised, train_mask)
    X, y, dates = create_sequences(close_scaled, df.index)
    logger.info(f"   sequências: X={X.shape}  y={y.shape}")
    
    # 4. Split train/val/test
    logger.info("[4/7] Split train/val/test...")
    train_x, train_y, train_d, val_x, val_y, val_d, test_x, test_y, test_d = split_train_val_test(X, y, dates)
    logger.info(f"   train: {train_x.shape}  val: {val_x.shape}  test: {test_x.shape}")
    
    # 5. Carregar modelo
    logger.info(f"[5/7] Carregando modelo {model_name}...")
    if model_name == "xlstm":
        checkpoint_path = ROOT / "xlstm_ts_full_checkpoint.pth"
        model, model_info = load_xlstm_model(checkpoint_path)
    elif model_name == "transformer":
        checkpoint_path = ROOT / "transformer_ts_full_checkpoint.pth"
        model, model_info = load_transformer_model(checkpoint_path)
    else:
        raise ValueError(f"Modelo desconhecido: {model_name}")
    
    # 6. Preparar background (do treino) e test subset
    logger.info("[6/7] Preparando background (do treino) e test subset...")
    background, test_subset, bg_indices = prepare_background_and_test(train_x, test_x)
    
    # 7. Calcular SHAP values
    logger.info("[7/7] Calculando SHAP values...")
    shap_values, expected_value = compute_shap_values(model, background, test_subset, model_name)
    
    # 8. Analisar importância temporal
    logger.info("Analisando importância temporal...")
    analysis = analyze_temporal_importance(shap_values)
    
    # 9. Gerar visualizações
    temporal_importance = np.array(analysis['temporal_importance'])
    generate_shap_plots(shap_values, temporal_importance, model_name, XAI_PLOTS_DIR)
    
    # 10. Salvar resultados
    results = {
        'model': model_name,
        'checkpoint': str(checkpoint_path),
        'device': str(DEVICE),
        'preprocessing': 'DWT causal (mode=zero, threshold treino) + MinMax (fit treino only)',
        'denoising_info': denoising_info,
        'model_info': model_info,
        'background_samples': len(background),
        'background_indices': bg_indices.tolist(),
        'test_samples': len(test_subset),
        'expected_value': float(expected_value),
        'shap_analysis': analysis,
    }
    
    output_json = XAI_DIR / f'shap_{model_name.lower().replace("-", "_")}.json'
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Resultados salvos: {output_json}")
    logger.info(f"Top timesteps: {analysis['top_timesteps'][:5]}")
    logger.info(f"Timesteps para 90% de importância: {analysis['n_timesteps_90pct']}")
    
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='xAI Analysis com SHAP')
    parser.add_argument('--model', type=str, choices=['xlstm', 'transformer', 'both'],
                        default='both', help='Modelo para analisar')
    args = parser.parse_args()
    
    logger.info("Iniciando xAI Analysis...")
    logger.info(f"Modelo(s): {args.model}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Background samples: {BACKGROUND_SAMPLES}")
    logger.info(f"Test samples: {TEST_SAMPLES}")
    
    if args.model == 'both':
        results = {}
        for model in ['xlstm', 'transformer']:
            try:
                results[model] = run_xai_for_model(model)
            except Exception as e:
                logger.error(f"Erro ao processar {model}: {e}")
                results[model] = {'error': str(e)}
    else:
        results = run_xai_for_model(args.model)
    
    logger.info("xAI Analysis concluído.")
    return results


if __name__ == "__main__":
    main()
