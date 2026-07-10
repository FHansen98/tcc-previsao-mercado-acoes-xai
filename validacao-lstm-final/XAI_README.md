# xAI Analysis — SHAP para xLSTM-TS e Transformer-TS

## Objetivo

Aplicar SHAP (SHapley Additive exPlanations) nos modelos treinados (xLSTM-TS e Transformer-TS) para interpretar quais timesteps passados mais influenciam as previsões de preço do S&P 500.

## Script Principal

**`src/xai_analysis.py`** — Script completo de análise de interpretabilidade.

## Dependências Necessárias

O script requer as seguintes bibliotecas:

- **torch** >= 2.0.0 (PyTorch)
- **xlstm** >= 1.0.0 (apenas para xLSTM-TS)
- **shap** >= 0.45.0 (biblioteca de interpretabilidade)
- **numpy**, **pandas**, **matplotlib**, **pywt**, **scikit-learn**

### Instalação

**Ambiente correto**: Os treinos de xLSTM-TS e Transformer-TS foram realizados no ambiente `.venv-xlstm` em `/home/ssdfhansen/Documentos/01_UnB/Matérias/ENS/TCC/validacao-bolsa/`. Este ambiente já possui todas as dependências necessárias para xAI.

**Executar no ambiente original**:
```bash
/home/ssdfhansen/Documentos/01_UnB/Matérias/ENS/TCC/validacao-bolsa/.venv-xlstm/bin/python \
  /home/ssdfhansen/Documentos/01_UnB/Matérias/ENS/TCC/validacao-lstm-final/src/xai_analysis.py \
  --model both
```

**Dependências já instaladas no ambiente .venv-xlstm**:
- torch 2.3.1+cpu
- shap 0.49.1
- xlstm 1.0.3
- numpy, pandas, matplotlib, pywt, scikit-learn

**Nota**: Não instale novas bibliotecas. O ambiente original já está configurado corretamente.

## Como Executar

**Importante**: Use sempre o ambiente `.venv-xlstm` onde os modelos foram treinados.

### Analisar apenas xLSTM-TS
```bash
/home/ssdfhansen/Documentos/01_UnB/Matérias/ENS/TCC/validacao-bolsa/.venv-xlstm/bin/python \
  /home/ssdfhansen/Documentos/01_UnB/Matérias/ENS/TCC/validacao-lstm-final/src/xai_analysis.py \
  --model xlstm
```

### Analisar apenas Transformer-TS
```bash
/home/ssdfhansen/Documentos/01_UnB/Matérias/ENS/TCC/validacao-bolsa/.venv-xlstm/bin/python \
  /home/ssdfhansen/Documentos/01_UnB/Matérias/ENS/TCC/validacao-lstm-final/src/xai_analysis.py \
  --model transformer
```

### Analisar ambos os modelos (padrão)
```bash
/home/ssdfhansen/Documentos/01_UnB/Matérias/ENS/TCC/validacao-bolsa/.venv-xlstm/bin/python \
  /home/ssdfhansen/Documentos/01_UnB/Matérias/ENS/TCC/validacao-lstm-final/src/xai_analysis.py \
  --model both
```

## Saídas Geradas

### Arquivos JSON (`results/xai/`)

- **`shap_xlstm.json`** — Métricas de interpretabilidade do xLSTM-TS
- **`shap_transformer.json`** — Métricas de interpretabilidade do Transformer-TS

Cada JSON contém:
- `temporal_importance`: Importância média por timestep (array de 150 valores)
- `top_timesteps`: Top 10 timesteps mais importantes
- `top_importance`: Valores de importância dos top timesteps
- `n_timesteps_90pct`: Quantos timesteps explicam 90% da importância
- `cumulative_importance`: Curva de importância acumulada

### Visualizações (`results/xai/plots/`)

- **`shap_xlstm.png`** — 4 painéis de visualização para xLSTM-TS
- **`shap_transformer.png`** — 4 painéis de visualização para Transformer-TS

Cada plot contém:
1. **Importância temporal** — Line plot de importância por timestep
2. **Heatmap** — SHAP values por amostra e timestep
3. **Top timesteps** — Bar chart dos 10 timesteps mais importantes
4. **Importância acumulada** — Curva de importância acumulada (90% threshold)

### Logs

- **`logs/xai_analysis.log`** — Log detalhado da execução

## Funcionamento do Script

### Pipeline de Execução

1. **Carregar dados** — `sp500_clean.csv` (2000-2024)
2. **Denoising** — DWT causal (mode='zero', Variante B)
3. **Normalização** — MinMaxScaler fit SOMENTE no treino
4. **Sequências** — Lookback=150 dias
5. **Split temporal** — Train (2000-2020), Val (2021-2022), Test (2023-2024)
6. **Carregar modelo** — Checkpoint treinado (`xlstm_ts_full_checkpoint.pth` ou `transformer_ts_full_checkpoint.pth`)
7. **Preparar background** — 100 amostras aleatórias do **TREINO** (nunca do teste)
8. **Calcular SHAP values** — Para 200 amostras do teste
9. **Analisar importância temporal** — Ranking de timesteps
10. **Gerar visualizações** — Plots e métricas

### Prevenção de Vazamento de Dados

**Regra fundamental**: Background dataset vem APENAS do treino.

```python
# ✅ CORRETO
background = train_x[np.random.choice(100, replace=False)]  # Do treino

# ❌ INCORRETO
background = test_x[np.random.choice(100, replace=False)]  # Do teste (vazamento!)
```

O script implementa essa regra automaticamente na função `prepare_background_and_test()`.

## Estrutura de Wrappers

### xLSTMWrapper

O xLSTM-TS tem 3 componentes separados:
- `xlstm_stack` — Stack de blocos xLSTM
- `input_projection` — Linear(1→64)
- `output_projection` — Linear(64→1)

O wrapper combina os 3 componentes em um único módulo PyTorch que SHAP entende:

```python
class xLSTMWrapper(nn.Module):
    def forward(self, x):
        x_proj = self.input_projection(x)
        xlstm_out = self.xlstm_stack(x_proj)
        return self.output_projection(xlstm_out[:, -1, :])
```

### TransformerWrapper

O Transformer-TS já é um módulo PyTorch, então o wrapper é simples:

```python
class TransformerWrapper(nn.Module):
    def forward(self, x):
        return self.model(x)
```

## Problema Atual: Disco Cheio

**Situação**: O disco local está 99% cheio (só 2.5GB livres), o que impede a instalação de torch (~2GB) e xlstm.

**Soluções**:

### Opção 1: Liberar espaço no disco local
```bash
# Verificar o que está ocupando espaço
du -sh /home/ssdfhansen/Documentos/01_UnB/Matérias/ENS/TCC/validacao-lstm-final/* | sort -hr

# Remover arquivos temporários ou grandes desnecessários
rm -rf /tmp/*
```

### Opção 2: Executar em outro ambiente
O script `xai_analysis.py` está pronto para rodar em qualquer ambiente com as dependências instaladas:

- **Servidor universitário** — Copie o diretório `validacao-lstm-final` e instale dependências
- **Google Colab** — Upload do script + dados + checkpoints
- **Outra máquina** — Se você tiver acesso a outro computador com mais espaço

### Opção 3: Usar ambiente conda (se disponível)
Se você tiver conda instalado, crie um ambiente separado:

```bash
conda create -n xai_env python=3.10
conda activate xai_env
pip install torch xlstm shap numpy pandas matplotlib pywt scikit-learn
python src/xai_analysis.py --model both
```

## Interpretação dos Resultados

### Temporal Importance

O script gera um ranking de timesteps (t-1 a t-150) por importância:

- **t-1** = Dia imediatamente anterior à previsão
- **t-2** = 2 dias antes
- ...
- **t-150** = 150 dias antes

**Esperado**: Timesteps recentes (t-1 a t-10) devem ter maior importância, com padrão decrescente.

### n_timesteps_90pct

Indica quantos dias passados são necessários para explicar 90% da previsão:

- Se `n_timesteps_90pct = 30`, então os 30 dias mais recentes explicam 90% da decisão do modelo
- Isso indica que o modelo foca principalmente em curto prazo

### Comparação entre Modelos

Compare os resultados de xLSTM-TS e Transformer-TS:

- **xLSTM-TS** pode ter memória mais longa (maior importância de timesteps remotos)
- **Transformer-TS** pode focar mais em timesteps recentes (attention local)
- Valores similares indicam convergência de explicações

## Integração com TCC

### Seção Sugerida no TCC

**Capítulo: Interpretabilidade com SHAP**

1. **Introdução à xAI**
   - Conceito de explainable AI
   - SHAP (SHapley Additive exPlanations)
   - DeepExplainer para modelos de deep learning

2. **Metodologia**
   - Background dataset (100 amostras do treino)
   - Cálculo de SHAP values (200 amostras do teste)
   - Prevenção de vazamento de dados

3. **Resultados — xLSTM-TS**
   - Importância temporal por timestep
   - Top timesteps mais importantes
   - Cumulative importance (quantos dias explicam 90%)

4. **Resultados — Transformer-TS**
   - Importância temporal por timestep
   - Comparação com xLSTM-TS
   - Attention weights vs SHAP values (se disponível)

5. **Discussão**
   - Convergência de explicações entre modelos
   - Timesteps consistentemente importantes
   - Implicações para trading (quais sinais priorizar)

## Tempo Estimado de Execução

| Tarefa | Tempo (CPU) | Tempo (GPU) |
|---|---|---|
| Carregar dados | 30s | 30s |
| Denoising + Normalização | 10s | 10s |
| Carregar modelo | 5s | 5s |
| Preparar background | 5s | 5s |
| Calcular SHAP (xLSTM) | 10-20 min | 2-5 min |
| Calcular SHAP (Transformer) | 10-20 min | 2-5 min |
| Gerar plots | 30s | 30s |
| **Total (1 modelo)** | **~15-25 min** | **~5-10 min** |
| **Total (ambos)** | **~30-50 min** | **~10-20 min** |

## Troubleshooting

### Erro: "torch não disponível"
**Causa**: torch não está instalado no ambiente.
**Solução**: `env/bin/python -m pip install torch`

### Erro: "xlstm não disponível"
**Causa**: xlstm não está instalado (apenas necessário para xLSTM-TS).
**Solução**: `env/bin/python -m pip install xlstm`

### Erro: "shap não disponível"
**Causa**: shap não está instalado.
**Solução**: `env/bin/python -m pip install shap`

### Erro: "Não há espaço disponível no dispositivo"
**Causa**: Disco cheio durante instalação.
**Solução**: Libere espaço ou use outro ambiente (ver seção "Problema Atual").

### Erro: "Checkpoint não encontrado"
**Causa**: Arquivo `.pth` não existe no diretório raiz.
**Solução**: Verifique se `xlstm_ts_full_checkpoint.pth` e `transformer_ts_full_checkpoint.pth` existem.

## Contato e Suporte

Para dúvidas ou problemas:
- Verifique o log: `logs/xai_analysis.log`
- Verifique o README do repositório external: `external/README.md`
- Consulte a documentação do SHAP: https://shap.readthedocs.io/
