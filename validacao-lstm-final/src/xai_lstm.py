#!/usr/bin/env python3
"""
Análise XAI para LSTM baseline (Variante B) usando SHAP.
Adaptado de xai_analysis.py para funcionar com TensorFlow/Keras.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

import shap

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent))
from lstm_base import (
    carregar_clean, preparar_features_de, standardize_train,
    make_seq, split_by_date, FEATURE_COLS, TRAIN_FIM, VAL_INI, TEST_INI,
    WINNING_CFG, build_model
)

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / 'results'
XAI_DIR = RESULTS_DIR / 'xai'
PLOTS_DIR = XAI_DIR / 'plots'

XAI_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def log(msg: str):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - {msg}")

def load_and_prepare_data():
    """Carrega e prepara dados usando Variante B (DWT denoising + Z-score)."""
    log("Carregando dados...")
    df_clean = carregar_clean()
    
    log("Preparando features (Variante B: DWT denoising + Z-score)...")
    X, y, diag = preparar_features_de(
        df_clean, denoise=True, label='Variante B'
    )
    
    log("Normalizando (Z-score calculado no treino)...")
    Xs, scaler_info = standardize_train(X)
    
    log("Criando sequências (lookback=20)...")
    X_all, y_all, ds_all = make_seq(Xs, y, WINNING_CFG['lookback'])
    
    log("Split train/val/test...")
    (Xtr, ytr, _), (Xva, yva, _), (Xte, yte, _) = split_by_date(X_all, y_all, ds_all)
    
    log(f"Train: {Xtr.shape}, Val: {Xva.shape}, Test: {Xte.shape}")
    
    return Xtr, Xva, Xte, ytr, yva, yte, diag, scaler_info

def train_or_load_model(Xtr, ytr, Xva, yva):
    """Treina ou carrega o modelo LSTM baseline."""
    checkpoint_path = BASE_DIR / 'lstm_baseline_checkpoint.keras'
    
    if checkpoint_path.exists():
        log(f"Carregando checkpoint existente: {checkpoint_path}")
        model = keras.models.load_model(checkpoint_path)
        return model, 0  # Epochs não disponível
    
    log("Treinando modelo LSTM baseline (Variante B)...")
    model = build_model(WINNING_CFG, Xtr.shape[2])
    
    cb = [keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=WINNING_CFG['patience'],
        restore_best_weights=True
    )]
    
    h = model.fit(
        Xtr, ytr.astype('float32'),
        validation_data=(Xva, yva.astype('float32')),
        epochs=WINNING_CFG['epochs'],
        batch_size=WINNING_CFG['batch'],
        verbose=1,
        callbacks=cb,
        shuffle=False
    )
    
    log(f"Salvando checkpoint: {checkpoint_path}")
    model.save(checkpoint_path)
    
    best_epoch = int(np.argmin(h.history['val_loss'])) + 1
    log(f"Best epoch: {best_epoch}")
    
    return model, best_epoch

def compute_shap_values(model, X_background, X_test):
    """Calcula SHAP values usando KernelExplainer (mais lento mas compatível)."""
    log("Calculando SHAP values para LSTM...")
    log("Usando KernelExplainer (model-agnostic)...")
    
    # KernelExplainer funciona com qualquer modelo, mas é mais lento
    n_background = 30
    n_test = 200
    
    X_bg = X_background[:n_background]
    X_te = X_test[:n_test]
    
    # Reshape para 2D (flatten timesteps e features) para KernelExplainer
    X_bg_flat = X_bg.reshape(X_bg.shape[0], -1)
    X_te_flat = X_te.reshape(X_te.shape[0], -1)
    
    def predict_wrapper(x):
        """Wrapper para reshaping predictions."""
        x_3d = x.reshape(x.shape[0], WINNING_CFG['lookback'], -1)
        return model.predict(x_3d, verbose=0).ravel()
    
    explainer = shap.KernelExplainer(predict_wrapper, X_bg_flat)
    # nsamples deve exceder o número de features (360) para evitar regressão degenerada
    # e valores SHAP numericamente instáveis
    shap_values = explainer.shap_values(X_te_flat, nsamples=500, l1_reg='num_features(20)')
    expected_value = explainer.expected_value
    
    # Reshape shap_values de volta para 3D
    shap_values_3d = shap_values.reshape(X_te.shape)
    
    log("SHAP values calculados com sucesso")
    return shap_values_3d, expected_value

def analyze_temporal_importance(shap_values, n_timesteps=20):
    """Analisa importância temporal média."""
    # shap_values tem shape (n_samples, n_timesteps, n_features)
    # Para LSTM baseline: (n_samples, 20, 18)
    
    # Agregar por timestep (média absoluta através das features)
    temporal_importance = np.mean(np.abs(shap_values), axis=(0, 2))
    
    # Ordenar timesteps por importância
    top_indices = np.argsort(temporal_importance)[::-1]
    top_importance = temporal_importance[top_indices]
    
    # Calcular timesteps necessários para 90% da importância
    sorted_importance = np.sort(temporal_importance)[::-1]
    cumsum = np.cumsum(sorted_importance)
    n_90pct = np.argmax(cumsum >= 0.9 * cumsum[-1]) + 1
    
    return {
        'temporal_importance': temporal_importance.tolist(),
        'top_timesteps': top_indices.tolist(),
        'top_importance': top_importance.tolist(),
        'n_timesteps_90pct': int(n_90pct)
    }

def generate_shap_plots(shap_values, X_test, temporal_importance, model_name):
    """Gera os 4 painéis de gráficos SHAP."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Importância temporal por timestep
    ax = axes[0, 0]
    timesteps = np.arange(len(temporal_importance['temporal_importance']))
    importance = temporal_importance['temporal_importance']
    ax.bar(timesteps, importance, color='steelblue', alpha=0.7)
    ax.set_xlabel('Timestep (dias atrás)', fontsize=10)
    ax.set_ylabel('Importância média (|SHAP|)', fontsize=10)
    ax.set_title(f'{model_name}: Importância Temporal por Timestep', fontsize=12, weight='bold')
    ax.grid(True, alpha=0.3)
    
    # 2. Heatmap de magnitude |SHAP| por timestep e amostra
    ax = axes[0, 1]
    n_samples_plot = min(200, shap_values.shape[0])
    shap_sample = shap_values[:n_samples_plot]
    # Magnitude por timestep: média absoluta através das features
    # (mesma métrica do gráfico de barras, garantindo coerência entre painéis)
    shap_mag = np.mean(np.abs(shap_sample), axis=2)
    vmax = np.percentile(shap_mag, 99)
    im = ax.imshow(shap_mag.T, aspect='auto', cmap='viridis', vmin=0, vmax=vmax)
    ax.set_xlabel('Amostra', fontsize=10)
    ax.set_ylabel('Timestep', fontsize=10)
    ax.set_title(f'{model_name}: Heatmap |SHAP| por timestep', fontsize=12, weight='bold')
    plt.colorbar(im, ax=ax, label='|SHAP| médio')
    
    # 3. Top 10 timesteps mais importantes
    ax = axes[1, 0]
    top_n = min(10, len(temporal_importance['top_timesteps']))
    top_indices = temporal_importance['top_timesteps'][:top_n]
    top_importance = temporal_importance['top_importance'][:top_n]
    ax.barh(range(top_n), top_importance[::-1], color='coral', alpha=0.7)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([f"T-{i}" for i in top_indices[::-1]])
    ax.set_xlabel('Importância média (|SHAP|)', fontsize=10)
    ax.set_title(f'{model_name}: Top {top_n} Timesteps', fontsize=12, weight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 4. Importância acumulada
    ax = axes[1, 1]
    sorted_importance = np.sort(temporal_importance['temporal_importance'])[::-1]
    cumsum = np.cumsum(sorted_importance)
    cumsum_pct = cumsum / cumsum[-1] * 100
    ax.plot(range(len(cumsum_pct)), cumsum_pct, 'o-', color='green', alpha=0.7)
    ax.axhline(y=90, color='red', linestyle='--', label='90%')
    ax.axvline(x=temporal_importance['n_timesteps_90pct'], color='red', linestyle='--')
    ax.set_xlabel('Timesteps ordenados por importância', fontsize=10)
    ax.set_ylabel('Importância acumulada (%)', fontsize=10)
    ax.set_title(f'{model_name}: Importância Acumulada', fontsize=12, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = PLOTS_DIR / f'shap_{model_name.lower()}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"Plot salvo: {output_path}")

def main():
    """Pipeline principal de XAI para LSTM baseline."""
    log("="*70)
    log("xAI Analysis — LSTM Baseline (Variante B)")
    log("="*70)
    
    # 1. Carregar e preparar dados
    Xtr, Xva, Xte, ytr, yva, yte, diag, scaler_info = load_and_prepare_data()
    
    # 2. Treinar ou carregar modelo
    model, best_epoch = train_or_load_model(Xtr, ytr, Xva, yva)
    
    # 3. Preparar background e test subset
    log("Preparando background (do treino) e test subset...")
    np.random.seed(42)  # Reprodutibilidade da amostragem
    n_background = 100
    n_test = 200
    
    background_indices = np.random.choice(len(Xtr), n_background, replace=False)
    X_background = Xtr[background_indices]
    
    test_indices = np.random.choice(len(Xte), n_test, replace=False)
    X_test_subset = Xte[test_indices]
    
    log(f"Background shape: {X_background.shape}")
    log(f"Test subset shape: {X_test_subset.shape}")
    
    # 4. Calcular SHAP values
    shap_values, expected_value = compute_shap_values(model, X_background, X_test_subset)
    
    # 5. Analisar importância temporal
    log("Analisando importância temporal...")
    temporal_analysis = analyze_temporal_importance(shap_values, n_timesteps=WINNING_CFG['lookback'])
    
    log(f"Top timesteps: {temporal_analysis['top_timesteps'][:5]}")
    log(f"Timesteps para 90% de importância: {temporal_analysis['n_timesteps_90pct']}")
    
    # 6. Gerar plots
    log("Gerando plots para LSTM...")
    generate_shap_plots(shap_values, X_test_subset, temporal_analysis, 'LSTM')
    
    # 7. Salvar resultados
    results = {
        'model': 'lstm_baseline',
        'variant': 'B',
        'checkpoint': str(BASE_DIR / 'lstm_baseline_checkpoint.keras'),
        'best_epoch': best_epoch,
        'preprocessing': diag,
        'scaler': scaler_info,
        'model_info': {
            'lookback': WINNING_CFG['lookback'],
            'units': WINNING_CFG['units'],
            'dropout': WINNING_CFG['dropout'],
            'l2': WINNING_CFG.get('l2', 0),
            'learning_rate': WINNING_CFG['lr'],
            'batch_size': WINNING_CFG['batch'],
            'epochs': WINNING_CFG['epochs'],
            'patience': WINNING_CFG['patience']
        },
        'background_samples': n_background,
        'background_indices': background_indices.tolist(),
        'test_samples': n_test,
        'expected_value': float(expected_value) if np.isscalar(expected_value) else expected_value.tolist(),
        'shap_analysis': temporal_analysis
    }
    
    output_json = XAI_DIR / 'shap_lstm.json'
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    log(f"Resultados salvos: {output_json}")
    log("="*70)
    log("xAI Analysis concluído.")

if __name__ == '__main__':
    main()
