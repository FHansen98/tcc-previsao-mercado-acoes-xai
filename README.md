# TCC - Previsão de Séries Temporais Financeiras com Deep Learning e xAI

Repositório do Trabalho de Conclusão de Curso sobre previsão direcional do índice S&P 500 utilizando modelos de Deep Learning (LSTM, xLSTM-TS, Transformer-TS) integrados a técnicas de Inteligência Artificial Explicável (SHAP).

## Estrutura do Repositório

### `template-latex-tcc/`
Arquivos LaTeX para geração do documento do TCC.

### `validacao-lstm-final/`
Projeto de experimentação com código fonte dos modelos (LSTM, xLSTM, Transformer), dados históricos, logs, resultados e documentação técnica (SHAP, conceitos, planejamento).

### `apresentacao_PG1/`
Arquivos da apresentação do TCC em slides em LaTeX Beamer.

### `correcoes/`
Documentos com correções e feedback dos professores da banca.

### Arquivos na Raiz

- **Dados e scripts**: `BVSP_dados_historicos.csv`, `baixar_bovespa.py`, `baixar-dados.md`
- **Documentação de estudo**: `analise_parametros_financeiros.md`, `comparacao.md`, `fichamento_artigos.md`, `comentarios_apresentacao.md`
- **Artigos da revisão bibliográfica**: `artigo1.pdf` a `artigo7.pdf` (artigos base para o projeto)
- **Versões do TCC**: `tcc*_Felipe_Hansen.pdf`
- **Apresentação**: `apresentacao_tcc.pdf`
- **Outros**: Instruções institucionais, tabela de hiperparâmetros

## Ambientes Virtuais

- **`validacao-bolsa/.venv/`**: LSTM baseline (TensorFlow/Keras)
- **`validacao-bolsa/.venv-xlstm/`**: Modelos SOTA (PyTorch + xlstm + shap)

## Como Compilar o TCC

```bash
cd template-latex-tcc
docker-compose up
```

## Como Compilar a Apresentação

```bash
cd apresentacao_PG1
docker compose up --build
```

## Como Executar os Experimentos

LSTM baseline:
```bash
cd validacao-lstm-final
source ../validacao-bolsa/.venv/bin/activate
python src/lstm_base.py
```

Modelos SOTA (xLSTM/Transformer):
```bash
cd validacao-lstm-final
source ../validacao-bolsa/.venv-xlstm/bin/python
python src/xlstm_ts.py
python src/transformer_ts.py
```

Análise SHAP:
```bash
python src/xai_analysis.py --model both
```

## Autor

Felipe Hansen - TCC Engenharia de Software, Universidade de Brasília (UnB)
