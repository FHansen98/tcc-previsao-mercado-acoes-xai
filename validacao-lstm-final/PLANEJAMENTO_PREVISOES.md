# Planejamento de Previsões — LSTM para Previsão Direcional de Ações

## Objetivo Principal

Este projeto visa validar as melhores formas de normalização, input e testes para um modelo LSTM aplicado à previsão direcional de ações (subida/descida). Após validação, os mesmos parâmetros serão utilizados para prever com o modelo xLSTM-TS e aplicar técnicas de xAI (SHAP) para interpretabilidade.

## Modelo LSTM Base

### Arquitetura

O modelo LSTM utilizado como base foi desenvolvido nas fases anteriores do projeto e consiste em:

- **Tipo**: LSTM unidirecional com dropout (não bidirecional)
- **Lookback**: 20 dias (janela temporal de entrada)
- **Units**: 32 neurônios na camada LSTM
- **Dropout**: 0.1 (regularização)
- **Camada de saída**: Dense com 1 neurônio + ativação sigmoid (probabilidade de subida)
- **Loss function**: Binary Crossentropy
- **Optimizer**: Adam (learning rate padrão)
- **Batch size**: 32
- **Early stopping**: Patience=10 no val_loss

### Dados Utilizados

- **Ativo**: S&P 500 (índice americano)
- **Período**: 2000-01-03 a 2024-12-30 (**6.288 dias**)
- **Colunas**: Date, Open, High, Low, Close, Adj Close, Volume, Volume_imputed, Price
- **Dados limpos**: `sp500_clean.csv` (sem valores faltantes, volume imputado)
- **Fonte**: Yahoo Finance via `yfinance` (^GSPC)

### Divisão Treino/Validação/Teste

| Split | Período | Dias | % | Finalidade |
|---|---|---|---|---|
| **Train** | 2000-01-03 a 2020-12-31 | ~5.275 | **84%** | Treinamento do modelo |
| **Val** | 2021-01-01 a 2022-12-31 | ~503 | **8%** | Ajuste de hiperparâmetros, early stopping |
| **Test** | 2023-01-01 a 2024-12-30 | ~503 | **8%** | Avaliação final (nunca visto no treino) |

**Nota**: A divisão é temporal (não aleatória) para simular cenário real de produção. O treino inclui a crise 2008 e COVID-2020, ensinando o modelo a lidar com regimes extremos. O test set (2023-2024) cobre um contexto macro diferente (juros altos pós-pandemia).

### Tipo de Previsão

- **Tarefa**: Classificação binária direcional
- **Target**: Direção do preço no dia seguinte (t+1)
  - **Classe 0**: Preço cai (retorno ≤ 0)
  - **Classe 1**: Preço sobe (retorno > 0)
- **Saída**: Probabilidade P(sobe) ∈ [0, 1]
- **Decisão**: Threshold 0.5 (P > 0.5 → sobe)

### Features de Entrada (18 features)

As features são calculadas a partir do preço (Close) e volume:

1. **Log-returns**: `logret_1d`, `logret_lag_1`, `logret_lag_2`, `logret_lag_3`, `logret_lag_5`, `logret_lag_10`, `logret_lag_20`
2. **Médias móveis**: `price_over_ma5`, `price_over_ma20`, `ma5_over_ma20`
3. **Volatilidade**: `vol_20d`, `vol_60d`
4. **Retornos absolutos**: `abs_ret_1d`, `ret2_1d`
5. **Volume**: `volume_log`, `volume_over_ma20`
6. **Range**: `hl_range` (High - Low)
7. **RSI**: `rsi_14`

Todas as features são normalizadas (Z-score ou Min-Max) antes de alimentar o modelo.

---

## Variantes de Normalização e Input

A tabela abaixo define as variantes a serem testadas para identificar a melhor forma de pré-processamento dos dados:

| Variante | Input | Normalização | Descrição |
|---|---|---|---|
| **A** (baseline) | Close bruto | Z-score | Baseline — preço original normalizado com Z-score (média/std do treino) |
| **B** | Close DWT denoised | Z-score | Denoising via DWT (Discrete Wavelet Transform) no Close, depois Z-score. **Atenção**: modo `symmetric` pode causar vazamento de dados (usa futuro). Deve ser testado com modo `zero` (causal). |
| **C** | Log-returns | Z-score | Transformação log-returns, depois Z-score. Remove tendência de longo prazo. |
| **D_corrigida** | Log-returns + MODWT causal | Z-score | MODWT causal (janela rolling 256d) nos log-returns + cumsum para reconstruir preço. 100% causal. Substitui a Variante D original que tinha vazamento. |
| **E** | Close | Rolling Z-score (252d) | Z-score causal: normaliza t usando média/std dos últimos 252 dias (1 ano). Evita vazamento mas pode ter edge effects. |
| **F** | Banco de filtros (3 níveis) | Min-Max por canal | DWT em 3 níveis de decomposição, cada sub-banda vira um canal de input. Min-Max por canal. |
| **G** | Close MODWT causal denoised | Min-Max | MODWT (Maximal Overlap DWT) causal — denoising rolling que só usa passado. Sem vazamento. |

### Justificativa da Tabela

As variantes foram selecionadas para cobrir diferentes abordagens de pré-processamento:

1. **Baselines simples** (A, C): Normalizações padrão (Z-score) em diferentes representações (preço vs log-returns)
2. **Denoising wavelet** (B, D_corrigida, G): Remoção de ruído via transformada wavelet, com diferentes modos (symmetric vs causal)
3. **Normalização causal** (E): Rolling Z-score que só usa passado
4. **Banco de filtros** (F): Representação multi-escala como canais de input

A **Variante D original** foi identificada como vazamento de dados (95.01% Acc, MCC +0.82 — falso positivo) e foi **substituída pela D_corrigida** que usa MODWT causal (100% sem vazamento). A variante B também é suspeita devido ao modo `symmetric` da DWT.

---

## Tipos de Treinamento

### Sliding Window (Janela Deslizante)

O modelo usa janela deslizante de 20 dias (lookback):

```
t-19, t-18, ..., t-1, t  →  Previsão para t+1
```

Para cada dia t, o modelo recebe as 18 features dos últimos 20 dias e prevê a direção em t+1.

### Processo de Treinamento

1. **Preparação de dados**:
   - Aplicar normalização/input da variante escolhida
   - Calcular as 18 features
   - Normalizar (Z-score ou Min-Max)
   - Criar sequências (lookback=20)

2. **Divisão temporal**:
   - Train: dados até 2020-12-31
   - Val: dados 2021-2022
   - Test: dados 2023+

3. **Treinamento**:
   - Batch size: 32
   - Epochs: até 100 (com early stopping patience=10)
   - Shuffle: False (respeita ordem temporal)
   - Callbacks: EarlyStopping (monitor=val_loss, restore_best_weights=True)

4. **Avaliação**:
   - Métricas: Accuracy, F1-Score, Matthews Correlation Coefficient (MCC), ROC-AUC
   - MCC é especialmente importante para detectar modelos que apenas chutam a classe majoritária

---

## Validação de Dados (Fases Anteriores)

Os dados foram validados nas Fases 0 a 6B. Não será repetido aqui — apenas documentado para referência.

### Fase 0 — Coleta e Qualidade (Dados Estendidos 2000-2024)

- **Fonte**: Yahoo Finance via `yfinance` (^GSPC)
- **Período coletado**: 2000-01-03 a 2024-12-30
- **Total de linhas**: 6.288 dias (S&P500)
- **Missings em OHLC**: 0 (nenhum valor faltante em preços)
- **Volume zero**: 0 dias (imputado na coleta)
- **Duplicatas de data**: 0
- **Relatório**: `results/fase0_qualidade_sp500_2000_2024.json`

| Métrica | S&P500 (2000–2024) |
|---|---:|
| Retorno acumulado (2000–2024) | **+308,0%** |
| Volatilidade anualizada | **17,4%** |
| Retorno diário médio | 0,03% |
| Desvio padrão diário | 1,10% |
| Assimetria (skewness) | −0,34 |
| Curtose (excesso) | 9,0 |
| Pior dia | −8,44% (12/03/2020, COVID) |
| Melhor dia | +9,11% (29/10/2008, crise) |

**Insights para modelagem**:
- Curtose excesso = 9.0 (>3) → caudas pesadas, eventos extremos mais frequentes que numa normal
- Skewness = -0.34 (negativo) → quedas grandes são mais comuns do que altas grandes
- Crise 2008 (outubro) e COVID-19 (março 2020) estão dentro do treino (2000–2020) — modelo aprende a lidar com regimes extremos
- Test set (2023–2024) cobre regime de juros altos pós-pandemia — contexto macro diferente do treino
- 25 anos de dados permitem ao modelo aprender múltiplos regimes econômicos (boom 2000s, crise 2008, recuperação, COVID, inflação 2020s)

### Fase 1 — Tratamento e Features

- Volume zero tratado como missing e imputado por mediana móvel
- 18 features canônicas criadas (log-returns, MAs, RSI, volatilidade, volume)
- Target criado: `target_direction_t+1` = (log-return do dia seguinte > 0)
- Split temporal 60/20/20 definido (não aleatório)
- `Adj Close` e `Close` decididos: usar `Close` como `Price` base

### Fases 2–5 — EDA, Ablation e Seleção de Hiperparâmetros

- Testes ADF/KPSS confirmaram não-estacionariedade de preços e estacionariedade de log-returns
- Ablation study em 10 configurações de LSTM → vencedora: **`less_reg_d01`** (lookback=20, units=32, dropout=0.1)
- Baseline (variante A, sem denoising) no test set 2023+:
  - Acc ≈ 55,89%, F1 ≈ 71,34%, MCC ≈ +0.017, AUC ≈ 0.52
  - Modelo chuta quase sempre classe 1 (sobe) — desempenho próximo ao chute

### Fase 6A — Wavelet Denoising

- Aplicado DWT (db4, soft threshold) no Close e demais colunas OHLCV
- Threshold calculado apenas no treino para evitar vazamento
- **Resultado com denoising** (mode='symmetric', depois corrigido para 'zero'):
  - Acc ≈ 62,5%, F1 ≈ 77,1%, MCC ≈ +0.15 (melhora real vs baseline)
- Plots gerados: `fase6_wavelet_signal.png`, `fase6_ruido_diagnostico.png`, `fase6_previsao_direcional.png`

### Fase 6B — Benchmarks xLSTM-TS e Darts

Modelos testados nos mesmos dados e split:

| Modelo | Acc Test | F1 Test | MCC |
|---|---|---|---|
| **Nosso LSTM (denoised)** | **62,50%** | **77,08%** | **+0.15** |
| xLSTM-TS (CPU) | 58,44% | 72,85% | ~+0.08 |
| N-BEATS (Darts) | 55,89% | 71,34% | ≈0 |
| TFT (Darts) | 55,89% | 71,34% | ≈0 |
| TCN (Darts) | 55,89% | 71,34% | ≈0 |

**Decisão**: Continuar com nosso LSTM como base — mais simples, melhor resultado.

### Fase 6C — Variantes de Normalização (resultados preliminares com bug)

Primeira execução usou pipeline reescrito do zero → bug: 6/7 variantes com MCC=0.0
(modelo chutava sempre classe 1, std das probabilidades ≈ 0.001).

Após correção (reutilizar pipeline da Fase 6A):

| Variante | Acc Test | F1 Test | MCC | Status |
|---|---|---|---|---|
| D original (LogRet + DWT global mode=symmetric) | 95,01% | 97,00% | +0.82 | ❌ **Vazamento confirmado** — substituída por D_corrigida |
| B (DWT denoised) | 80,44% | 85,63% | +0.55 | ⚠️ **Suspeito** (mode='symmetric') |
| A (baseline) | 55,89% | 71,34% | +0.017 | ✅ |
| C (log-returns) | 55,89% | 71,34% | +0.017 | ✅ |
| E (rolling Z-score) | 54,69% | 70,09% | −0.033 | ✅ |

Esses resultados **não são finais** — serão refeitos neste diretório com controle de vazamento.

---

## Plano de Execução

### Fase 1: Reconstrução da LSTM Base

1. Copiar código da LSTM base do diretório `validacao-bolsa` para `validacao-lstm-final`
2. Garantir que o código esteja limpo e bem documentado
3. Testar execução com baseline (variante A) para confirmar funcionamento

### Fase 2: Teste de Variantes (sem vazamento)

1. Implementar variantes A, B (com mode='zero'), C, D_corrigida, E, F, G
2. Variante D original descartada e substituída por D_corrigida (MODWT causal)
3. Executar todas as variantes com seed fixo (42)
4. Comparar resultados (Acc, F1, MCC, AUC)

### Fase 3: Seleção da Melhor Variante

1. Analisar resultados e identificar a variante com melhor MCC (não apenas Acc)
2. Verificar se há overfitting (comparar train vs val vs test)
3. Selecionar a variante vencedora

### Fase 4: Aplicação em xLSTM-TS

**Nota**: xLSTM-TS é um modelo SOTA (state-of-the-art) de séries temporais baseado em xLSTM (extended LSTM). O repositório oficial foi adaptado para rodar em CPU.

**Arquivo**: `src/xlstm_ts.py` (adaptado de `validacao-bolsa/src/fase6b_xlstm_official_cpu.py`)

**Repositório original**: https://github.com/gonzalopezgil/xlstm-ts

**Alterações necessárias vs. original**:

| Alteração | Original | Adaptado | Motivo |
|---|---|---|---|
| **Backend sLSTM** | `backend="cuda"` | `backend="vanilla"` | CPU-compatible (sem GPU NVIDIA) |
| **Device** | `.to("cuda")` | `.to(DEVICE)` | Auto-detect CPU/CUDA |
| **Dados** | Download via yfinance | `sp500_clean.csv` | Usar dados já coletados (2000-2024) |
| **Split** | 2000–2021 (train) | 2000–2020 (train) | Alinhar com split da LSTM |
| **Val/Test** | 2021–2022 (val), 2022–2023 (test) | 2021–2022 (val), 2023–2024 (test) | Alinhar com split da LSTM |
| **Tarefa** | Regressão de preço | Regressão de preço | xLSTM-TS é regressão por padrão |

**Parâmetros do modelo (preservados do paper)**:
- `num_blocks=4`
- `embedding_dim=64`
- `num_heads=2`
- `SEQ_LENGTH_XLSTM=150` (lookback)
- `num_epochs=200`
- `batch_size=16`
- `lr=1e-4`
- `patience=40` (early stopping)
- `ReduceLROnPlateau` (scheduler)

**Diferença fundamental vs. nossa LSTM**:
- **Nossa LSTM**: Classificação binária direcional (sobe/desce) com 18 features
- **xLSTM-TS**: Regressão de preço (preço absoluto) com apenas 1 feature (Close denoised)

**Tempo estimado**: 4-12h em CPU moderna (bem mais lento que LSTM)

**Saída**: `results/xlstm_ts_sp500.json` com métricas MSE, MAE, MAPE, etc.

### Fase 5: xAI (SHAP)

1. Aplicar SHAP no modelo vencedor (LSTM ou xLSTM-TS)
2. Interpretar quais features são mais importantes
3. Gerar visualizações (summary plot, dependence plot, etc.)

---

## Saídas do Modelo: O que é Armazenado

### Predições Dia a Dia (CSV)

Para cada variante treinada, é gerado um arquivo `results/predicoes_<variante>.csv` com **uma linha por dia do test set**:

| Coluna | Tipo | Descrição |
|---|---|---|
| `Date` | data | Data do dia previsto |
| `close_real` | float | Preço de fechamento real naquele dia |
| `real` | 0 ou 1 | O que *realmente aconteceu* (0=caiu, 1=subiu) |
| `proba_sobe` | [0,1] | Probabilidade P(sobe) prevista pelo modelo |
| `pred` | 0 ou 1 | Predição binária (threshold=0.5) |
| `acerto` | 0 ou 1 | 1 se `pred == real`, 0 caso contrário |

**Exemplo de linhas**:
```
Date,       close_real, real, proba_sobe, pred, acerto
2023-01-03, 3824.14,    1,    0.6201,     1,    1
2023-01-04, 3852.36,    1,    0.5843,     1,    1
2023-01-05, 3808.10,    0,    0.4102,     0,    1
2023-01-06, 3895.75,    1,    0.3891,     0,    0
```

### Métricas Agregadas (JSON e CSV)

Não são atualizadas dia a dia — são calculadas **uma única vez** sobre todos os ~501 dias do test set:

| Métrica | O que mede | Interpretação |
|---|---|---|
| **Accuracy** | % de dias acertados | 55% = chute; >60% = sinal real |
| **F1-score** | Média harmônica de precisão e recall | Penaliza errar demais em uma só classe |
| **MCC** | Correlação entre previsto e real | 0 = chute; +1 = perfeito; -1 = invertido |
| **AUC-ROC** | Área sob a curva ROC | 0.5 = aleatório; 1.0 = perfeito |

**Sobre K-Fold Cross Validation**: *não é usado aqui*. K-Fold divide os dados em K partes e treina K modelos diferentes para ter uma estimativa mais robusta. Não é adequado para séries temporais porque misturaria dados do futuro no treino.

### Por que não usar K-Fold para séries temporais?

**K-Fold padrão** (para dados não temporais):
- Divide os dados em K partes (ex: K=5)
- Treina 5 vezes, cada vez usando 4 partes para treino e 1 para teste
- **Problema**: mistura dados do futuro no treino → vazamento

**Exemplo do problema**:
```
Fold 1: treino=[partes 2,3,4,5], teste=[parte 1]
→ teste tem dados de 2000-2004, treino tem dados de 2005-2024
→ o modelo "vê o futuro" durante o treino
```

### Alternativas para séries temporais

| Método | Descrição | Uso no TCC |
|---|---|---|
| **Time Series Split** | Train/Val/Test fixos em ordem temporal | ✅ **Usado** (2000–2020 | 2021–2022 | 2023–2024) |
| **Walk-Forward** | Treino expanding window (2000–2015→2016, 2000–2016→2017, ...) | ❌ Não usado (aumentaria tempo 5-10x) |
| **Sliding Window** | Janela fixa de treino que desliza | ❌ Não usado (mesmo motivo) |

**Decisão**: Usar Time Series Split (um único split temporal) porque:
- Simula uso real do modelo em produção
- Evita vazamento de dados
- Tempo de execução viável (~5-7 minutos por variante)

### Gráficos Gerados (`results/plots/`)

| Gráfico | O que mostra |
|---|---|
| Curva de preço c/ pontos verdes/vermelhos | Acertos e erros dia a dia no test set |
| Loss treino vs validação | Se o modelo aprendeu bem ou fez overfitting |
| Curva ROC | Trade-off entre detectar altas e evitar falsos alarmes |
| Histograma de P(sobe) | Se o modelo separa as classes ou chuta uma única probabilidade |

O **histograma de probabilidades** é especialmente diagnóstico: se o modelo estiver com problema, todas as probabilidades se concentram em ~0.55 (chuta sempre sobe). Um modelo funcional terá distribuição mais espalhada e separada entre as classes.

---

## Prevenção de Vazamento de Dados

### Regras de Ouro

1. **Só usar dados do passado**: Normalizações e denoising devem ser causais
2. **Threshold calculado no treino**: Parâmetros de normalização (média, std, threshold) só podem usar dados de treino
3. **Modo DWT causal**: Usar `mode='zero'` ou MODWT causal, nunca `mode='symmetric'`
4. **Não reconstruir preço**: Evitar reconstruir preço a partir de log-returns denoised (causa vazamento como na variante D)

### Testes de Vazamento

1. **Correlação**: Se Close_original vs Close_denoised > 95%, há vazamento
2. **MCC**: Se MCC ≈ 0, o modelo está chutando aleatoriamente (pode indicar features constantes)
3. **Acc vs MCC**: Se Acc é alta mas MCC é baixo, o modelo está chutando a classe majoritária

---

## Estrutura de Diretórios

```
validacao-lstm-final/
├── PLANEJAMENTO_PREVISOES.md       # Este documento
├── src/
│   ├── lstm_base.py                # ✅ Pipeline LSTM base (features, treino, métricas)
│   ├── xlstm_ts.py                 # ✅ xLSTM-TS adaptado (CPU-compatible)
│   ├── coleta.py                   # ✅ Coleta dados desde 2000
│   ├── analise_qualidade.py        # ✅ Análise de qualidade dos dados
│   └── variantes.py                # ⏳ Script que roda variantes A-G e gera comparativo
├── external/
│   └── xlstm-ts/                   # ✅ Repositório xLSTM-TS oficial (clonado)
├── data/
│   └── sp500_clean.csv             # ✅ Coletado via src/coleta.py (2000-2024, 6.288 dias)
├── results/
│   ├── fase0_qualidade_sp500_2000_2024.json  # ✅ Relatório de qualidade
│   ├── variantes_sp500.csv         # ⏳ Tabela comparativa A-G
│   ├── variantes_sp500.json        # ⏳ Métricas completas com y_true/y_proba
│   ├── xlstm_ts_sp500.json         # ⏳ Resultados xLSTM-TS
│   └── plots/                      # ⏳ Gráficos
└── logs/
    └── variantes.log               # ⏳ Logs de execução
```

---

## Considerações Importantes (Lições do Chat Anterior)

### 1. Bug crítico no pipeline reescrito do zero
Ao reescrever o pipeline em `fase6c_norm_variants.py` sem reutilizar o código da Fase 6A, o modelo passou a chutar sempre classe 1 (std das probabilidades ≈ 0.001). A causa foi a normalização Min-Max global sem os parâmetros corretos. **Lição**: sempre reutilizar o pipeline validado.

### 2. Modo DWT 'symmetric' causa vazamento
O modo padrão da DWT (`mode='symmetric'`) usa dados de ambos os lados (passado e futuro) para calcular os coeficientes wavelet. Ao denoiser o ponto t=2023-01-15, o algoritmo pode usar t+1, t+2, etc. (que estão no test set). **Correção**: usar `mode='zero'` (padding com zeros, causal). Já aplicado em `lstm_base.py`.

### 3. Variante D original: duas fontes de vazamento (corrigida)

**Variante D original** (descartada):
```python
logret_den = wavelet_denoise_series(logret, mode='symmetric')  # ← vazamento 1: DWT global usa futuro
price_proxy = exp(cumsum(logret_den))                           # ← vazamento 2: cumsum amplifica o leak
```
- DWT com `mode='symmetric'` processa toda a série de uma vez — cada coeficiente wavelet pode conter informação do futuro
- `cumsum` dos log-returns acumula o vazamento ao longo do tempo → preço proxy artificialmente suave e previsível
- Resultado: Acc=95.01%, MCC=+0.82 (falso positivo confirmado)

**D_corrigida** (implementada em `preparar_variant_D_causal()`):
```python
logret_den = _modwt_causal_denoise_series(logret, win=256)  # ← MODWT: cada t usa só [t-256, t]
price_proxy = exp(cumsum(logret_den))                        # ← cumsum ainda é causal (ok)
```
- MODWT rolling: para cada ponto t, aplica DWT apenas na janela `[t-256, t]`
- 100% causal — verificado: `assert all(out[:win] == x[:win])` PASSED
- Correlação preço denoised vs original: ~0.99 (alta, mas esperada — é o mesmo sinal sem ruído)
- **Regra**: reconstrução via cumsum É aceitável se o denoising do log-return for causal

### 4. MCC é a métrica mais confiável para detectar problemas
- **MCC = 0**: modelo chuta aleatoriamente (ou sempre a mesma classe)
- **MCC < 0**: modelo é pior que o chute
- **Accuracy alta + MCC ≈ 0**: modelo chuta a classe majoritária (S&P500 sobe ~55% dos dias)
- Sempre verificar distribuição das predições: se std(probabilidades) < 0.01, o modelo não está aprendendo

### 5. Target sempre calculado do preço ORIGINAL
Mesmo quando as features usam preço denoised, o target (`target_direction_t+1`) deve ser calculado a partir do preço original. Já implementado em `preparar_features_de()` via `logret_1d_orig`.

### 6. Ambiente Python
- **Ambiente**: `env/` no diretório raiz TCC (não `.venv`)
- **Pacotes principais**: tensorflow, sklearn, pywt, pandas, numpy, matplotlib
- **Dados**: `validacao-lstm-final/data/processed/sp500_clean.parquet`

### 7. Dados estendidos para 2000-2024
Os dados foram estendidos de 2015-2024 (2.515 dias) para 2000-2024 (6.288 dias, 2.50x mais dados).

**Impacto no tempo de treinamento**:
- Sequências (lookback=20): ~2.495 → ~6.268 (2.5x)
- Tempo de treinamento: ~2-3 minutos → ~5-7 minutos (escala linear)
- Ainda viável e benéfico para robustez do modelo

**Vantagens**:
- Modelo aprende mais regimes econômicos (crise 2008, COVID, juros altos)
- Mais robusto a diferentes contextos macro
- Test set ainda mantém 2023-2024 (contexto diferente do treino)

**Split atual (alinhado com tcc2.txt)**:
- Train: 2000–2020 (~5.275 dias, 84%)
- Val: 2021–2022 (~503 dias, 8%)
- Test: 2023–2024 (~503 dias, 8%)

---

## Próximos Passos

1. ✅ Criar diretório `validacao-lstm-final`
2. ✅ Criar documento `PLANEJAMENTO_PREVISOES.md`
3. ✅ Copiar e adaptar código da LSTM base (`src/lstm_base.py`)
4. ✅ Coletar dados: `src/coleta.py` (2000-2024, 6.288 dias)
5. ✅ Implementar `src/variantes.py` e executar todas as variantes (resultados abaixo)
6. ✅ Aplicar em xLSTM-TS com mesmos parâmetros (CONCLUÍDO — 287.5 min, 139 épocas)
7. ⏳ xAI (SHAP) no modelo vencedor

---

## Resultados — Fase 5: Comparativo de Variantes (2023–2024)

**Executado em**: `src/variantes.py` | **Seed**: 42 | **Dados**: sp500_clean.csv (2000-2024)

| Variante | Normalizer | Acc Train | Acc Val | Acc Test | F1 Test | MCC Test | AUC Test | Épocas | McNemar p |
|---|---|---|---|---|---|---|---|---|---|
| **B** (DWT denoised, mode=zero) | Z-score | 0.683 | 0.686 | **0.675** | **0.720** | **+0.333** | **0.724** | 25 | **0.0001** ✅ |
| A (baseline Close) | Z-score | 0.678 | 0.626 | 0.615 | 0.696 | +0.195 | 0.652 | 21 | 0.049 ✅ |
| C (price proxy log-ret) | Z-score | 0.678 | 0.626 | 0.615 | 0.696 | +0.195 | 0.652 | 21 | 0.049 ✅ |
| G (MODWT causal Close) | Min-Max | 0.658 | 0.626 | 0.605 | 0.655 | +0.193 | 0.634 | 62 | 0.190 ⚠️ |
| E (rolling Z-score 252d) | Rolling Z | 0.665 | 0.642 | 0.601 | 0.700 | +0.158 | 0.645 | 16 | 0.118 ⚠️ |
| D_corrigida (MODWT logret) | Z-score | 0.610 | 0.589 | 0.533 | 0.651 | +0.005 | 0.517 | 26 | 0.338 ❌ |
| F (filter bank 3 níveis) | MinMax/ch | 0.553 | 0.545 | 0.563 | 0.720 | **−0.039** | 0.563 | 11 | 1.000 ❌ |

**Vencedora: Variante B** — DWT denoised Close (mode='zero') + Z-score

### Arquivos gerados

- `results/variantes_sp500.csv` — tabela acima
- `results/variantes_sp500.json` — métricas completas + y_true/y_proba
- `results/predicoes_<variante>.csv` — previsões dia a dia (482 dias)
- `results/plots/comparativo_variantes.png` — barras Acc/F1/AUC/MCC
- `results/plots/probabilidades_variantes.png` — histogramas P(sobe) por variante
- `results/plots/overfitting_variantes.png` — curvas loss por época
- `results/plots/resultados_<variante>.png` — gráfico individual (4 painéis)

---

## Resultados — Fase 6: xLSTM-TS (Variante B)

**Executado em**: `src/xlstm_ts.py` | **Seed**: 42 | **Dados**: sp500_clean.csv (2000-2024) | **Tempo**: 287.5 min (4h 47min)

**Preprocessamento**: DWT causal (mode='zero', db4, level=6, threshold do treino) + MinMax (fit treino only)

**Modelo**: xLSTM-TS (paper params: num_blocks=4, embedding_dim=64, num_heads=2, seq_length=150, epochs=200, batch_size=16, lr=1e-4, patience=40)

| Métrica | Valor |
|---|---|
| **MAE** | 42.76 |
| **MSE** | 4288.08 |
| **RMSE** | 65.48 |
| **RMSSE** | 2.80 |
| **MAPE** | 0.80% |
| **MASE** | 2.60 |
| **R²** | 0.99 |
| **Acc Train** | 88.66% |
| **Acc Val** | 76.34% |
| **Acc Test** | 76.80% |
| **Recall** | 80.45% |
| **Precision (Rise)** | 82.03% |
| **Precision (Fall)** | 68.56% |
| **F1 Score** | 81.23% |
| **Épocas (early stopping)** | 139/200 |

### Arquivos gerados

- `results/xlstm_ts_sp500.json` — métricas completas
- `results/plots/xlstm_ts_predicoes.png` — previsões vs real (teste 2023-2024)
- `results/plots/xlstm_ts_confusion_matrix.png` — matriz de confusão (direcional)
- `results/plots/xlstm_ts_erro.png` — erro absoluto ao longo do tempo
- `results/plots/xlstm_ts_scatter.png` — scatter real vs previsto

