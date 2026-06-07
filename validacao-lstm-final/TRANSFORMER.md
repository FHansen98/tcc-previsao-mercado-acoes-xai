# Adaptando seu projeto LSTM → Transformer para Previsão de Bolsa (Keras/TensorFlow)

O documento usa PyTorch/HuggingFace, mas a **arquitetura e os conceitos são idênticos**. Vou mapear cada etapa do PDF para equivalentes em Keras/TensorFlow.

---

## 🗺️ Roteiro Completo

---

### ETAPA 1 — Instalação e Importações

```python
pip install tensorflow scikit-learn matplotlib pandas yfinance
```

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import yfinance as yf
```

---

### ETAPA 2 — Download e Exploração dos Dados

> *Equivalente à Seção 2 do PDF: "Download e Exploração do Dataset IMDB"*

No PDF, o dataset é carregado e inspecionado antes de qualquer processamento. Faça o mesmo com seus dados de bolsa:

```python
# Baixar dados históricos
ticker = "PETR4.SA"  # ou qualquer ativo
df = yf.download(ticker, start="2018-01-01", end="2024-01-01")

# Inspeção (equivalente ao print de estatísticas do PDF)
print(df.shape)
print(df.describe())
print(df.isnull().sum())

# Visualização exploratória
plt.figure(figsize=(12, 4))
plt.plot(df['Close'])
plt.title(f'Preço de Fechamento — {ticker}')
plt.xlabel('Data')
plt.ylabel('Preço (R$)')
plt.savefig('dataset_eda.png', dpi=120)
plt.show()
```

---

### ETAPA 3 — Pré-processamento e Criação de Janelas (equivale à Tokenização)

> *Equivalente à Seção 3 do PDF: "Tokenização, Dataset Customizado e DataLoader"*

No PDF, o tokenizador converte texto em sequências de IDs com comprimento fixo (`max_length=256`). Para séries temporais, o equivalente é a **janela deslizante**, que transforma a série em sequências de tamanho fixo.

```python
# ── Hiperparâmetros globais (equivalente ao bloco de configuração do PDF) ──
SEQ_LEN      = 60      # equivalente ao max_length=256 do PDF
BATCH_SIZE   = 32      # idêntico ao PDF
EPOCHS       = 30
LEARNING_RATE = 1e-4
N_HEADS      = 4       # cabeças de atenção (PDF usa 12 no DistilBERT)
D_MODEL      = 64      # dimensão do embedding (PDF usa 768)
N_LAYERS     = 2       # camadas transformer (PDF usa 6 no DistilBERT)
DROPOUT_RATE = 0.1
FEATURES     = ['Close', 'Volume', 'High', 'Low', 'Open']

# ── Normalização (equivalente ao padding/truncation do PDF) ────────────
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[FEATURES])

# ── Função de janela deslizante (equivalente ao IMDBDataset.__getitem__) ──
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])          # janela de entrada
        y.append(data[i + seq_len][0])            # próximo valor de Close
    return np.array(X), np.array(y)

X, y = create_sequences(data_scaled, SEQ_LEN)

# ── Divisão treino / validação / teste (mesmo padrão do PDF) ──────────
n = len(X)
n_train = int(n * 0.70)
n_val   = int(n * 0.15)

X_train, y_train = X[:n_train], y[:n_train]
X_val,   y_val   = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
X_test,  y_test  = X[n_train+n_val:], y[n_train+n_val:]

print(f"Treino: {X_train.shape} | Validação: {X_val.shape} | Teste: {X_test.shape}")

# ── tf.data.Dataset (equivalente ao DataLoader do PDF) ────────────────
train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

val_ds   = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
            .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

test_ds  = (tf.data.Dataset.from_tensor_slices((X_test, y_test))
            .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
```

> 💡 **Analogia com o PDF:** assim como o `attention_mask` indica quais tokens são reais, aqui todas as posições da janela são válidas — não há padding porque as janelas têm sempre tamanho fixo.

---

### ETAPA 4 — Construção do Modelo Transformer

> *Equivalente à Seção 4 do PDF: "Criação do Modelo Pré-Treinado"*

No PDF é usado o DistilBERT com uma cabeça linear de classificação. Aqui construímos um Transformer **do zero** com uma cabeça de regressão — a lógica é idêntica.

```python
# ── Bloco de Atenção Multi-Cabeça (núcleo do Transformer) ─────────────
def transformer_encoder_block(inputs, d_model, n_heads, dropout_rate):
    """
    Equivalente a uma camada do DistilBertModel do PDF.
    Contém: Multi-Head Attention + Add&Norm + FFN + Add&Norm
    """
    # 1. Multi-Head Self-Attention
    attn_output = layers.MultiHeadAttention(
        num_heads=n_heads,
        key_dim=d_model // n_heads,   # dimensão por cabeça
        dropout=dropout_rate
    )(inputs, inputs)                  # query = key = value (self-attention)

    # 2. Add & Norm (residual connection)
    x = layers.LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    # 3. Feed-Forward Network (FFN)
    ffn = layers.Dense(d_model * 4, activation='relu')(x)
    ffn = layers.Dropout(dropout_rate)(ffn)
    ffn = layers.Dense(d_model)(ffn)

    # 4. Add & Norm
    x = layers.LayerNormalization(epsilon=1e-6)(x + ffn)
    return x


# ── Modelo completo (equivalente ao AutoModelForSequenceClassification) ──
def build_transformer_model(seq_len, n_features, d_model, n_heads,
                             n_layers, dropout_rate):
    inputs = keras.Input(shape=(seq_len, n_features))

    # Projeção de entrada para d_model (equivalente ao embedding do BERT)
    x = layers.Dense(d_model)(inputs)

    # Positional Encoding (informa a ordem temporal ao modelo)
    positions = tf.range(start=0, limit=seq_len, delta=1)
    pos_embedding = layers.Embedding(input_dim=seq_len, output_dim=d_model)(positions)
    x = x + pos_embedding

    x = layers.Dropout(dropout_rate)(x)

    # Empilhar N blocos encoder (PDF usa 6 camadas no DistilBERT)
    for _ in range(n_layers):
        x = transformer_encoder_block(x, d_model, n_heads, dropout_rate)

    # Cabeça de regressão sobre o último token (equivalente ao [CLS] do PDF)
    x = x[:, -1, :]                          # pega o último passo temporal
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    output = layers.Dense(1)(x)              # previsão do próximo preço

    model = keras.Model(inputs=inputs, outputs=output)
    return model


model = build_transformer_model(
    seq_len=SEQ_LEN,
    n_features=len(FEATURES),
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    dropout_rate=DROPOUT_RATE
)

model.summary()
```

---

### ETAPA 5 — Otimizador, Scheduler e Loss

> *Equivalente à Seção 5 do PDF: "Otimizador, Scheduler e Função de Perda"*

O PDF usa **AdamW + scheduler linear com warmup**. Replicamos exatamente isso:

```python
# ── Scheduler com warmup (idêntico ao do PDF, adaptado para Keras) ────
total_steps   = (len(X_train) // BATCH_SIZE) * EPOCHS
warmup_steps  = int(total_steps * 0.1)   # 10% para warmup (mesmo do PDF)

print(f"Total de passos : {total_steps}")
print(f"Passos de warmup: {warmup_steps}")

class WarmupLinearDecay(keras.optimizers.schedules.LearningRateSchedule):
    """Equivalente ao get_linear_schedule_with_warmup do HuggingFace."""
    def __init__(self, peak_lr, warmup_steps, total_steps):
        self.peak_lr      = peak_lr
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps

    def __call__(self, step):
        step    = tf.cast(step, tf.float32)
        warmup  = step / tf.cast(self.warmup_steps, tf.float32)
        decay   = (self.total_steps - step) / tf.cast(
                   self.total_steps - self.warmup_steps, tf.float32)
        lr      = self.peak_lr * tf.minimum(warmup, decay)
        return tf.maximum(lr, 0.0)

lr_schedule = WarmupLinearDecay(LEARNING_RATE, warmup_steps, total_steps)

# AdamW (equivalente direto ao do PDF, weight_decay=0.01)
optimizer = keras.optimizers.AdamW(
    learning_rate=lr_schedule,
    weight_decay=0.01
)

# Compilar: MSE para regressão (análogo ao CrossEntropy do PDF para classificação)
model.compile(
    optimizer=optimizer,
    loss='mse',
    metrics=['mae']
)
```

---

### ETAPA 6, 7 e 8 — Loop de Treinamento Completo

> *Equivalente às Seções 6, 7 e 8 do PDF*

O PDF implementa manualmente o loop PyTorch. No Keras, o `model.fit()` encapsula tudo isso, incluindo checkpointing:

```python
# ── Callbacks (equivalente ao checkpointing manual do PDF) ────────────
callbacks = [
    # Salva o melhor modelo (val_loss), igual ao torch.save do PDF
    keras.callbacks.ModelCheckpoint(
        filepath='melhor_modelo.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    # Early stopping para evitar overfitting (mencionado no PDF seção 7)
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        verbose=1
    )
]

# ── Treinamento (equivalente ao loop de épocas da Seção 8 do PDF) ─────
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)
```

---

### ETAPA 9 — Avaliação Final

> *Equivalente à Seção 9 do PDF: "Avaliação Final no Conjunto de Teste"*

```python
# Carregar melhor modelo (equivalente ao model.load_state_dict do PDF)
model = keras.models.load_model('melhor_modelo.keras')

# Previsões no conjunto de teste
y_pred_scaled = model.predict(X_test)

# Desnormalizar (inverter o MinMaxScaler)
def desnormalizar(valores_scaled, scaler, n_features):
    dummy = np.zeros((len(valores_scaled), n_features))
    dummy[:, 0] = valores_scaled.flatten()
    return scaler.inverse_transform(dummy)[:, 0]

y_test_real = desnormalizar(y_test,       scaler, len(FEATURES))
y_pred_real = desnormalizar(y_pred_scaled, scaler, len(FEATURES))

# Métricas (equivalente ao classification_report do PDF)
mae  = mean_absolute_error(y_test_real, y_pred_real)
rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
mape = np.mean(np.abs((y_test_real - y_pred_real) / y_test_real)) * 100

print(f"MAE  : R$ {mae:.2f}")
print(f"RMSE : R$ {rmse:.2f}")
print(f"MAPE : {mape:.2f}%")
```

---

### ETAPA 10 — Visualização dos Resultados

> *Equivalente à Seção 10 do PDF*

```python
fig, axes = plt.subplots(1, 2, figsize=(16, 4))

# Curvas de treino (idêntico ao do PDF)
axes[0].plot(history.history['loss'],     label='Treino',    color='steelblue')
axes[0].plot(history.history['val_loss'], label='Validação', color='tomato', linestyle='--')
axes[0].set_title('Curva de Perda (MSE)')
axes[0].set_xlabel('Época')
axes[0].legend()

# Previsão vs Real
axes[1].plot(y_test_real,  label='Real',     color='steelblue')
axes[1].plot(y_pred_real,  label='Previsto', color='tomato', linestyle='--')
axes[1].set_title('Previsão vs Valor Real')
axes[1].set_xlabel('Dias')
axes[1].set_ylabel('Preço (R$)')
axes[1].legend()

plt.tight_layout()
plt.savefig('resultados_treinamento.png', dpi=120)
plt.show()
```

---

### ETAPA 11 — Inferência

> *Equivalente à Seção 11 do PDF: "Inferência com Frases Novas"*

```python
def prever_preco(df_recente, model, scaler, seq_len, features):
    """
    Equivalente à função prever_sentimento() do PDF.
    Recebe os últimos N dias e retorna a previsão do próximo fechamento.
    """
    dados = scaler.transform(df_recente[features].values[-seq_len:])
    X_input = dados[np.newaxis, ...]          # shape: (1, seq_len, n_features)
    pred_scaled = model.predict(X_input, verbose=0)
    return desnormalizar(pred_scaled, scaler, len(features))[0]

preco_amanha = prever_preco(df, model, scaler, SEQ_LEN, FEATURES)
print(f"Previsão do próximo fechamento: R$ {preco_amanha:.2f}")
```

---

## 📊 Tabela de Correspondência PDF → Seu Projeto

| Seção do PDF | Conceito | Adaptação para Bolsa (Keras) |
|---|---|---|
| Tokenizador + `max_length` | Converte texto em sequência fixa | Janela deslizante de `SEQ_LEN` dias |
| `attention_mask` | Marca tokens reais vs padding | Não necessário (janelas sempre completas) |
| `[CLS]` token | Representação global da sequência | Último passo temporal `x[:, -1, :]` |
| `IMDBDataset` | Dataset customizado PyTorch | `tf.data.Dataset` |
| `DataLoader` | Batching e shuffling | `.batch().shuffle().prefetch()` |
| `AutoModelForSequenceClassification` | Encoder + cabeça final | `build_transformer_model()` |
| `AdamW` + warmup scheduler | Otimização estável | `keras.optimizers.AdamW` + `WarmupLinearDecay` |
| `CrossEntropyLoss` | Loss para classificação | `MSE` para regressão |
| `torch.save` / `load_state_dict` | Checkpointing | `ModelCheckpoint` callback |
| `torch.no_grad()` + `model.eval()` | Modo inferência | `model.predict()` (automático no Keras) |
| `classification_report` | Métricas finais | MAE, RMSE, MAPE |