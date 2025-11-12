## Artigo 1: Um Método Algorítmico para Operações na Bolsa de Valores Baseado em Ensembles de Redes Neurais para Modelar e Prever os Movimentos dos Mercados de Ações 
O trabalho propõe dois ensembles de redes neurais: o ensemble moderado (composto por duas redes neurais, exigindo concordância para operar) e o ensemble agressivo (composto por duas redes neurais e o indicador técnico SAR, utilizando um módulo de votação quando retorna uma previsão).

Em ambos os ensembles, as redes neurais são do tipo Multilayer Perceptron (MLP) com arquitetura feed forward.

O treinamento foi supervisionado, utilizando o algoritmo Resilient Propagation (RProp) (em vez do backpropagation tradicional).

Os ensembles foram avaliados por índice de acertos e ganho de capital em simulações de operações. O ensemble agressivo obteve o maior lucro total e os melhores resultados de classificação na maioria das vezes, enquanto o moderado teve lucro em 100% das séries temporais avaliadas.

---

## Artigo 2: Estudo da Aplicação de Redes Neurais Artificiais para Predição de Séries Temporais Financeiras 
A pesquisa propõe e avalia um método de Ensemble de Redes Neurais Artificiais que consiste em calcular a média simples dos resultados previstos por três redes neurais distintas.

Foram utilizadas três arquiteturas de redes neurais: Multilayer Perceptron (MLP), rede neural Auto-Regressiva com Entradas Exógenas (NARX), e rede neural com Long Short-Term Memory (LSTM).

O treinamento foi realizado com a base de dados do Ibovespa e utilizou-se o método de otimização Adam com 100 épocas.

A avaliação dos modelos foi feita através de métricas de desempenho de erro: MSE, RMSE e MAPE, além da precisão do movimento (acerto na direção "Alta" ou "Baixa"). O Ensemble teve os melhores resultados nas métricas de erro, mas alcançou apenas 70% de acerto no movimento, enquanto as redes MLP e NARX, que obtiveram 80% de acerto.

---

## Artigo 3: Uma Avaliação Sistemática de Técnicas de Aprendizado de Máquina Baseadas em Ensemble para Previsão de Índices do Mercado de Ações Usando Séries Temporais Financeiras 
O trabalho compara várias arquiteturas de ensemble classificadas em: Ensemble Learning (Bagging e Stacking), Sistema Híbrido Residual, Decomposição (CEEMDAN), Otimização (Algoritmo Genético - AG) e Completo (Decomposição + Otimização).

Os modelos base para as abordagens de ensemble foram: CART (Árvore de Classificação e Regressão), MLP (Multilayer Perceptron), e SVR (Regressão por Vetores de Suporte).

Foi utilizado treinamento supervisionado com um protocolo para minimizar tendências: divisão 70% treino/30% teste, normalização MinMax, *cross-validation* com janela deslizante.

O trabalho focou na previsão de valores futuros (regressão) para os índices IBOVESPA e S&P 500. A avaliação foi feita com métricas de erro (MSE, RMSE, MAE, MAPE) e uma métrica de Custo-Benefício (relacionada ao Instructions Retired da CPU). Os resultados mostraram que o modelo CART teve desempenho superior ou estatisticamente equivalente às abordagens de ensemble na métrica de Custo-Benefício.

---

## Artigo 4: Uso de Redes Neurais Recorrentes para Previsão de Séries Temporais Financeiras 
Este trabalho não utiliza uma abordagem de ensemble, focando na aplicação e avaliação de um modelo de rede neural recorrente do tipo Long Short-Term Memory (LSTM).

O modelo principal é uma rede Long Short-Term Memory (LSTM). Para comparação foram utilizados Multilayer Perceptron (MLP) e Random Forest (RF).

O treinamento foi supervisionado, utilizando dados de preços históricos e indicadores de análise. O modelo utilizou uma janela deslizante de 10 entradas anteriores.

O objetivo era a previsão de tendências (classificação binária: alta/não alta). A avaliação do modelo foi realizada com métricas de acerto (Acurácia, Precisão, Revocação, Medida F1) e análise financeira (retorno percentual, lucro por operação). Os resultados da rede LSTM apresentaram uma acurácia média de até 55,9% e foram considerados estatisticamente. Resumidamente, o LSTM obteve retornos positivos em todos os ativos e apresentou menor risco.

---

## Artigo 5: Explainable Artificial Intelligence (XAI) in Predicting Stock Market Crashes: A Case Study of the Tehran Stock Exchange

O trabalho utiliza o XGBoost (Extreme Gradient Boosting), um algoritmo de *ensemble* altamente eficiente baseado em uma coleção de árvores de decisão impulsionadas por gradiente. A principal contribuição metodológica é a integração deste modelo com técnicas de Inteligência Artificial Explicável (XAI), como SHAP, LIME e Permutation Feature Importance para interpretar suas previsões e abrir a "caixa preta".

O modelo principal é o XGBoost, que é um algoritmo de *Machine Learning* e *ensemble* de árvores, e não uma rede neural artificial (como MLP ou LSTM). O modelo foi escolhido por sua alta *performance* e capacidade de lidar com dados estruturados.

Classificação binária supervisionada para prever a probabilidade de um *crash* de mercado. O evento *crash* foi definido como uma queda de mais de 10% no índice TEPIX (Bolsa de Valores de Teerã - TSE) dentro dos 21 dias de negociação subsequentes. Foram aplicadas estratégias para lidar com o desequilíbrio de classes (apenas 7,96% de eventos *crash*), incluindo SMOTE no conjunto de treinamento e o ajuste do parâmetro Scale Position Weight do XGBoost.

O modelo XGBoost alcançou uma acurácia geral de 97,25% no conjunto de teste. Para a classe crítica *Crash*, obteve um Recall (Sensibilidade) de 0.8846 e Precision de 0.7931 (F1-Score de 0.8364). Através das técnicas de XAI, o estudo revelou que a volatilidade de mercado (especialmente `Volatility_20D`), os níveis de preço (High, Open) e o Volume-Weighted Average Price (VWAP) foram os principais fatores (features) para determinar a probabilidade de *crash*. O XAI forneceu *insights* economicamente significativos, expondo as relações não lineares que impulsionam o risco de mercado.


---

## Artigo 6: Explainable AI (XAI) models applied to planning in financial markets

O trabalho utiliza a abordagem Gradient Boosting Decision Trees (GBDT), um método de *ensemble* que supera outros métodos de *Machine Learning* (ML) em termos de acurácia em problemas de classificação com conjuntos de dados pequenos e desbalanceados. O modelo específico usado foi o LightGBM. O GBDT é um candidato ideal para investigar a qualificação de regimes para mercados de ações.

O principal modelo de previsão é o GBDT (LightGBM). Para fins de comparação, foram avaliados outros modelos de ML:

- Deep LSTM (rede neural de duas camadas com LSTM e camadas densas).
- Deep FC (rede neural *fully connected* com três camadas).
- RBF SVM (Máquina de Vetores de Suporte com Kernel de Função de Base Radial).
- Random Forest (RF).

Tipo de Treinamento: Classificação binária supervisionada para prever crises de mercado no índice S&P 500. Um regime de crise é definido como uma ocorrência de retorno do índice abaixo do percentil histórico de 5% em um horizonte de 15 dias. O modelo foi treinado em mais de **150 *features*** (técnicas, fundamentais e macroeconômicas).

Resultado e Análise: O modelo GBDT com seleção de *features* (GBDT FS) foi superior em todas as métricas em comparação com os outros modelos testados. A análise se concentrou em explicar as decisões do modelo usando os valores SHAP (SHapley Additive exPlanation), fornecendo uma interpretação local e global das previsões. Essa análise de XAI revelou que o risco de *crash* é impulsionado por uma mistura de *features* pró-cíclicas e contra-cíclicas, destacando o papel preditivo contrarian do setor de tecnologia (*tech equity sector*) durante a crise de março de 2020.

---

## Artigo 7: An Interpretable Framework for Stock Market Forecasting using Long Short-Term Memory (LSTM) Networks with SHAP - Explainable AI (XAI) Method

Tipo de Ensemble: Este trabalho foca na aplicação de um modelo *single* Long Short-Term Memory (LSTM) com a integração de uma ferramenta de interpretabilidade SHAP, e não utiliza uma arquitetura de *ensemble* para previsão.

Redes Utilizadas: Foi empregada uma arquitetura de Stacked LSTM (LSTM Empilhada). O modelo consiste em três camadas LSTM (cada uma com 50 unidades) seguidas por uma camada Densa (*Dense layer*) com um único neurônio para a previsão final do preço.

Tipo de Treinamento: Regressão supervisionada para estimar os preços futuros de ações do HDFC Bank (ticker: HDFCBANK.NS). O *dataset* (de 2020 a 2025) foi pré-processado usando Min-Max Scaling para normalização. Foi utilizado um método de janela deslizante com um período de *look-back* de 100 dias. O modelo foi treinado com o otimizador Adam e a função de perda Mean Squared Error (MSE).

Resultado e Análise: O modelo demonstrou forte desempenho preditivo, alcançando uma R² (coeficiente de determinação) de 0.9489 e um RMSE de 0.0246. A Integração do SHAP (XAI) foi a chave para a análise, proporcionando interpretabilidade ao identificar a contribuição de cada passo de tempo no resultado previsto. O SHAP revelou que os dias mais recentes (por exemplo, 2 e 3 dias antes da previsão) geralmente tinham maior impacto na previsão do preço futuro.


# Comparação entre  os artigos

| Artigo | Classes/Índices de Classificação | Alertas/Momentos de Risco | Ganho de Capital (Valor ou Aumento) | Maior Ganho | Método de Avaliação Principal | Dados Utilizados para Análise |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **artigo1** | **Alta/Queda** da série temporal no próximo período. | **Sinais de Compra/Venda** para o próximo período. O ensemble moderado tem uma saída "não sabe" para incerteza. | **Aumento** na maioria dos casos (lucro em 89% dos casos avaliados). O ensemble moderado obteve lucro em 100% das séries temporais testadas. | Ensemble Agressivo (no geral), com um **lucro total de \$188k** (\$100k capital inicial por ação) para ambos os mercados. Lucro de **64%** em 166 dias (SWN) e até **7%** em 192 períodos de 15 min (BBSE3). | Previsão como problema de classificação (subir/descer), índice de acertos (acurácia), e variação de capital (lucro/prejuízo) de operações simuladas. | Mercados de ações **norte-americano (S&P 500)** e **brasileiro (Bovespa)**. Ações específicas: S&P 500, AA, BAC, C, F, FCX, GE, JPM, SWN, BOVA11, BBAS3, BBDC4, BBSE3, BRFS3, BVMF3, ITSA4, PETR4, VALE5. |
| **artigo2** | Previsão dos valores (Regressão), classificando em **Movimento do índice** (Alta ou Baixa) para o índice Bovespa. | Não especificado (foco na previsão do valor para "um-passo-a-frente"). | Não avaliado (foco nas métricas de erro e precisão do movimento). | MLP e NARX obtiveram a maior precisão no movimento (**80% de acerto**). Ensemble com o melhor MAPE (**0,54**) e RMSE (**503,49**). | MSE, RMSE, MAPE e Precisão Movimento (acerto na direção Alta/Baixa). | Séries temporais do **Índice Bovespa (Ibovespa)** e a cotação do **Dólar**. Período: 04 de janeiro de 2010 a 28 de dezembro de 2017. |
| **artigo3** | **Previsão de valores futuros** (Regressão) dos índices acionários (IBOVESPA e S&P 500). | **Não avaliado** (foco na regressão/previsão de índices, não em decisões de compra/venda). | O foco é no **trade-off** entre métricas de erro (RMSE) e Custo-Benefício (eficiência). O modelo **Bagging CART** foi o melhor em custo-benefício para o IBOVESPA, e **CART single** para o S&P 500. | O modelo **Decomposição CART** foi o melhor nas métricas de erro tradicionais para o IBOVESPA, e **Sistema Híbrido CART** para o S&P 500. O modelo **CART (single)** apresentou desempenho superior ou equivalente em custo-benefício em ambos os mercados. | MSE, RMSE, MAE, MAPE, e **Análise de Custo-Benefício** (usando *Instructions Retired* da CPU). Também utiliza testes de hipótese (Wilcoxon, Friedman, Nemenyi). | Índices do mercado de ações: **IBOVESPA** (emergente) e **S&P 500** (desenvolvido). Período: 1º de janeiro de 2009 a 30 de dezembro de 2019. |
| **artigo4** | **Movimento de preço** (Subir/Não subir) no período seguinte (15 minutos). | **Sinais de Compra/Venda** para o período seguinte (operação de compra/venda em t e fechada em t+1). | **Positivo** para todos os ativos testados (retorno financeiro simulado). | Redes **LSTM** (o modelo proposto) apresentou retornos médios por operação melhores na maior parte dos casos e um bom indicativo de baixo risco. Acurácia média de até **55,9%**. | Acurácia, Precisão, Revocação, Medida F1 (métricas de classificação) e **Retornos Financeiros** em comparação com estratégias de investimento simples (Buy-and-hold, Estratégia ingênua, etc.). | Ativos da bolsa de valores: **BOVA11, BBDC4, CIEL3, ITUB4, PETR4**. Granularidade: **15 minutos**. Dados históricos de preço e volume (candles) do ano de 2014. |
| **artigo5** | **Crash Event** ($>10\%$ de queda no índice TEPIX em 21 dias). | **Sinais de Crash** (probabilidade de crash). | Não avaliado (o foco é na precisão preditiva e interpretabilidade do risco). | F1-Score de **0.8364** (Threshold=0.5) para a classe 'Crash'. Modelo XGboost com forte acurácia preditiva. | Precisão, Recall, F1-Score, ROC AUC, PR AUC (foco na classe minoritária), e interpretações baseadas em XAI (SHAP, LIME, Permutation Feature Importance). | Índice geral da **Bolsa de Valores de Teerã (TSE)**, conhecido como **TEPIX**. Período: Março de 2018 até Maio de 2025. Inclui **79 features** de Análise Técnica (RSI, MACD, Volatility\_20D, VWAP, etc.). |
| **artigo6** | O modelo classifica o mercado em dois regimes: **"Normal"** ou **"Crise"**. O Regime de Crise é definido pela ocorrência de retorno do índice **abaixo do percentil histórico de 5%** em um horizonte de **15 dias**. | **Probabilidade de Crise** (crise/normal). O XAI revela *features* pró-cíclicas e contra-cíclicas que indicam o risco. | Não avaliado (o foco é na precisão de classificação e XAI). | GBDT FS **AUC de 0.83** (Superior ao Deep LSTM FS AUC de 0.74). | **Performance de Classificação** (Acurácia, F1-Score, AUC) e **Interpretabilidade (XAI)**, usando **Shapley values (SHAP)**. | **S&P 500 futures prices**. Mais de **150 features** (Técnicas, Fundamentais, Macroeconômicas, Risco, etc.). |
| **artigo7** | O objetivo é a **Previsão de valores futuros** (Regressão) do preço de fechamento da ação. O modelo foca em prever o preço exato para o horizonte de 10 e 30 dias. | Não avaliado diretamente, mas o **SHAP** fornece *insights* sobre a **influência temporal do risco** (quais dias passados mais influenciam a previsão). | Não avaliado (foco nas métricas de erro). | **RMSE de 0.0246** e **R² de 0.9489**. | **Métricas de Erro de Regressão** (RMSE, MAE, MSE) e o **Coeficiente de Determinação (R²)**. A qualidade é validada pela **Interpretabilidade (XAI)**, usando **SHAP**. | Preços históricos de fechamento da ação **HDFC Bank** (ticker: HDFCBANK.NS). |



### Resumos Detalhados por Artigo (Foco em Avaliação e Classificação)

| Artigo | Explicação Detalhada das Classes/Índices de Classificação | Explicação Detalhada do Método de Avaliação Principal |
| :--- | :--- | :--- |
| **artigo1** | A classificação fundamental é binária: **Alta** ou **Queda** da série temporal no próximo período. O sistema Ensemble Moderado introduz uma terceira classe implícita: **"Não Sabe"**, acionada por discordância ou incerteza entre as redes, evitando uma operação. Os alvos são movimentos de preço de ações específicas. | O modelo é avaliado pela **Acurácia de Classificação** (acerto na direção) e, crucialmente, pela **Rentabilidade Financeira**. É realizada uma **simulação de operações (paper trading)**, onde a variação do capital do investidor é acompanhada após a aplicação dos sinais de Compra/Venda gerados. O desempenho é comparado a estratégias como Buy-and-Hold e Estratégia Trivial. |
| **artigo2** | O foco primário é a **previsão de valores futuros** (regressão) para o índice Bovespa. A classificação derivada é o **Movimento do índice** (**Alta ou Baixa**) para "um-passo-a-frente". | O estudo utiliza métricas de erro tradicionais de **Regressão** (MSE, RMSE, MAPE) para quantificar a proximidade dos valores previstos em relação aos valores reais. Além disso, emprega uma métrica de **Classificação** chamada **Precisão do Movimento**, que avalia o percentual de acerto na direção (Alta/Baixa), independentemente da magnitude do erro no valor. |
| **artigo3** | O objetivo é a **Previsão de valores futuros** (Regressão) dos índices acionários (IBOVESPA e S&P 500). A comparação de desempenho se concentra nas categorias de modelos (Bagging, Decomposição, etc.) usadas para otimizar essa previsão. | Adota uma abordagem de **Avaliação Sistemática** com métricas de erro de **Regressão** (MSE, RMSE, MAE, MAPE) para medir a acurácia da previsão. O diferencial é a **Análise de Custo-Benefício**, que usa o custo computacional (*Instructions Retired* da CPU) em conjunto com o erro, buscando o **trade-off** ideal entre acurácia e eficiência. Testes estatísticos (*Wilcoxon*, *Friedman*, *Nemenyi*) validam a significância dos resultados. |
| **artigo4** | Classifica o movimento de preço como **Subir** ou **Não subir** no período seguinte (intervalo de 15 minutos), tratando a tarefa como um problema de classificação binária no curtíssimo prazo. O sinal resultante é usado diretamente para gerar operações de compra/venda. | A avaliação é dupla: **performance de Classificação** (Acurácia, Precisão, Revocação, Medida F1) e **Rentabilidade Financeira** simulada (Retornos Financeiros). O desempenho é comparado a métodos tradicionais (MLP, Random Forest) e estratégias passivas (Buy-and-hold). |
| **artigo5** | A principal classe é o **Crash Event** (**1**), definido como uma queda superior a **10%** no índice TEPIX dentro de um horizonte de **21 dias**. O modelo classifica cada dia como propenso a um crash (sinal de risco) ou não. | Focado na avaliação de **modelos de classificação para eventos raros e de alto impacto (Crash)**. Utiliza métricas que priorizam a classe minoritária, como Precisão, Recall, F1-Score, e as áreas sob a curva ROC (ROC AUC) e PR (PR AUC). A qualidade também é medida pela **Interpretabilidade (XAI)**, usando SHAP e LIME para explicar as previsões de risco. |
| **artigo6** | O modelo classifica o mercado em dois regimes: **"Normal"** ou **"Crise"**. O Regime de Crise é definido pela ocorrência de retorno do índice **abaixo do percentil histórico de 5%** em um horizonte de 15 dias. | O método avalia a **performance de Classificação** usando métricas como Acurácia, Precisão, Recall, F1-Score e **AUC** (Area Under the Curve). O foco principal é a **Interpretabilidade (XAI)**, usando **Shapley values (SHAP)** para fornecer explicações locais e globais das previsões de crise. |
| **artigo7** | O objetivo é a **Previsão de valores futuros** (Regressão) do preço de fechamento da ação. O modelo foca em prever o preço exato para o horizonte de 10 e 30 dias. | O estudo utiliza métricas de erro de **Regressão** (RMSE, MAE, MSE) e o **Coeficiente de Determinação (R²)** para medir a acurácia. A qualidade é validada pela **Interpretabilidade (XAI)**, usando **SHAP** para entender a contribuição de cada *time step* na previsão. |

## Previsão de Valores Futuros (Regressão)

O objetivo da regressão é apontar o valor exato do índice ou preço no futuro, e não apenas a quantidade da variação.

 - Saída da Regressão: O modelo retorna um valor numérico contínuo. Por exemplo, se o índice Bovespa está em 130.000 pontos hoje, a rede pode prever que ele estará em 130.750 pontos amanhã.

 - Não é uma "Classe": O output não é uma etiqueta categórica ("Alta" ou "Queda"), como na classificação. É um número real.


| Característica	| Regressão (Artigos 2, 3)	| Classificação (Artigos 1, 4, 5) |
| :--- | :--- | :--- |
Output da Rede	| Um número real (valor exato)	| Uma etiqueta ou classe (ex: 1 ou 0)
O que Preveem	| Qual será o preço futuro (ex: 130.750)	| Qual será a direção (ex: Subirá ou Cairá)
Medida de Erro	| MSE, RMSE, MAPE (médias de erro do valor)	| Acurácia, F1-Score (percentual de acerto da direção)
Uso em Operações	| Mais útil para precificar derivativos ou estimar o tamanho do movimento.	| Mais útil para gerar sinais binários de Compra/Venda.

## Ferramentas de Avaliação

| Artigo | Ferramentas/Frameworks Utilizados | Tipo de Rede ou Aplicação da Ferramenta |
| :--- | :--- | :--- |
| **artigo1** | **Encog** (framework) | Utilizado para implementar Redes Neurais **MLP** (MultiLayer Perceptron) do tipo Feed Forward. Adotou o algoritmo de treinamento *Resilient Propagation*. |
| **artigo2** | **Python**, **Keras** (framework), **TensorFlow** (biblioteca de back-end) | Keras/TensorFlow foram a interface e o motor para a construção e treinamento das redes **MLP**, **NARX** (Autorregressiva com entradas exógenas) e **LSTM** (Long Short-Term Memory). |
| **artigo3** | **Scikit-learn**, **pmdarima**, **EMD-signal** | **Scikit-learn** foi a base para a construção dos modelos de Machine Learning (MLP, SVR, CART) e das estruturas de **Ensemble** (Bagging, Stacking, Sistemas Híbridos). **pmdarima** foi usado para o modelo estatístico **ARIMA**. |
| **artigo4** | **Keras**, **TensorFlow**, **TA-Lib** | **Keras/TensorFlow** foram usados para construir o modelo principal de **Redes Neurais Recorrentes LSTM**. **TA-Lib** (*Technical Analysis Library*) foi a ferramenta utilizada para a engenharia de recursos (*feature engineering*), gerando os indicadores de análise técnica (MACD, RSI, etc.). |
| **artigo5** | **XGBoost**, **SHAP**, **LIME**, **matplotlib**, **finpy\_tse** | **XGBoost** é o modelo de **Ensemble de Árvores** (*Gradient Boosting*) utilizado para a previsão de *crash* (risco). **SHAP** e **LIME** são ferramentas de **XAI (Inteligência Artificial Explicável)**, usadas para interpretar as previsões e a contribuição das *features* (análise de risco). |
| **artigo6** | **LightGBM**, **Shapley values (SHAP)**, **TensorFlow/Keras** | **LightGBM** é o modelo de **GBDT** (*Gradient Boosting Decision Trees*) principal. **SHAP** (valores Shapley) é a ferramenta de **XAI** utilizada para fornecer uma compreensão global e explicações locais das decisões do modelo. **TensorFlow/Keras** foram usados para implementar os modelos de **Deep Learning** (Deep FC e Deep LSTM) de comparação. |
| **artigo7** | **Keras**, **TensorFlow**, **SHAP**, **Adam Optimizer** | O modelo principal é uma **Rede Neural LSTM Empilhada** (*Stacked LSTM*), implementada via **Keras/TensorFlow**. O **SHAP** é a ferramenta de **XAI** usada para interpretar a contribuição de cada *time step* (dia) na previsão de preço. |

## Escolha e Análise dos Modelos de Previsão

A escolha do modelo (Rede Neural, Ensemble de Árvores, e Machine Learning) em cada artigo foi motivada por um objetivo específico. A tabela abaixo lista os motivos da escolha, bem como as vantagens e desvantagens citadas nos artigos:

| Artigo | Modelo Escolhido | Motivo da Escolha (Decisão) | Vantagens (Citadas nos Artigos) | Desvantagens (Citadas nos Artigos) |
| :--- | :--- | :--- | :--- | :--- |
| **artigo1** | **Ensembles de Redes Neurais** (composta por MLP) | Redes Neurais (RN) do tipo perceptron de multicamadas (MLP) entregam **melhores resultados** na previsão de séries temporais do que outras técnicas de IA. Ensembles fornecem **melhor generalização** e **maior segurança** contra previsões erradas. | Superam o desempenho de RNs sozinhas. A redundância de várias redes diminui o erro e aumenta a segurança do resultado final. | Aumento no número de elementos do *ensemble* pode causar mais discordâncias e, consequentemente, mais saídas "não sabe". |
| **artigo2** | **MLP, NARX, LSTM** e **Ensemble** (Média) | A escolha visava **comparar** o desempenho de diferentes arquiteturas (simples, recorrente e profunda) e avaliar a eficácia do método **Ensemble** sobre o índice Bovespa. | **MLP** é um método simples em termos de complexidade computacional. **LSTM** é capaz de armazenar informações por longos períodos. **NARX** é capaz de mapear sistemas dinâmicos tipicamente não-lineares. | **MLP/NARX** tiveram 80% de acerto na direção, enquanto o **Ensemble** teve apenas 70%. Redes Neurais (RNAs) são complexas de modelar e prever. |
| **artigo3** | **MLP, SVR, CART, ARIMA** (e seus Ensembles) | Seleção baseada na **alta recorrência** e status de **algoritmos clássicos** na literatura sobre séries temporais financeiras, permitindo uma comparação abrangente e padronizada. | **MLP/SVR** podem lidar com dados não lineares. **CART** é eficaz na captura de relações não lineares. **ARIMA** é um modelo estatístico linear frequentemente citado na literatura. | Aumento da complexidade (*ensemble*) nem sempre gera ganho significativo no resultado final (trade-off). **CART** é suscetível a *overfitting*. **SVR** exige seleção cuidadosa de *kernel* e parâmetros. |
| **artigo4** | **Redes Neurais Recorrentes (LSTM)** | Proposta de que redes recorrentes são mais apropriadas para séries temporais por terem **memória de curto prazo**. | Melhor capacidade de lidar com sequências de dados longas, distinguindo ocorrências recentes e distantes, e ignorando memórias irrelevantes. | O problema de previsão de séries temporais financeiras é um desafio gigantesco, sendo considerado caótico e dinâmico, com baixa relação sinal/ruído. |
| **artigo5** | **XGBoost** (Ensemble de Árvores) | Escolhido por seu **histórico de alta performance preditiva** em domínios financeiros e por ser adequado para interpretação via XAI (especificamente TreeSHAP). | Altamente eficiente, escalável e robusto. Possui mecanismos intrínsecos de regularização (L1 e L2) para controlar *overfitting*. | É um modelo complexo de "caixa preta" (*black box*), dificultando a compreensão do processo interno de tomada de decisão, o que limita a confiança e a auditoria sem o uso de XAI. |
| **artigo6** | **GBDT** (*Gradient Boosting Decision Trees*) | Metodologia adequada para identificar regimes de mercado. É o método mais adequado para problemas de classificação em *datasets* **pequenos** e **desbalanceados** (classes imbalanced). | Demonstra acurácia superior sobre outros métodos de ML, como *Deep Learning* e RBF SVM. Permite treinamento rápido com *leaf-wise tree growth*. | É propenso a *overfitting* em aplicações de regressão. |
| **artigo7** | **Redes Neurais Recorrentes (LSTM)** | A rede LSTM é ideal por sua capacidade de lidar com **sequências de dados** e **dependências de longo prazo**. O estudo combina LSTM com **XAI** para criar uma estrutura interpretável. | Redes LSTM são capazes de reter informações cruciais por longas durações, superando muitos métodos tradicionais em precisão e robustez. | São frequentemente criticadas por serem modelos **"black-box"** que carecem de interpretabilidade. |

#### Classificação de Modelos de Machine Learning e Redes Neurais
A classificação dos modelos de Machine Learning (ML) se baseia na arquitetura interna e na metodologia matemática que eles utilizam:

 - Modelos Clássicos/Tradicionais (ou ML "Shallow"): São modelos que operam com métodos estatísticos, geometria (fronteiras) ou regras sequenciais (árvores). Eles não dependem de múltiplas camadas ocultas e geralmente exigem que o engenheiro de dados realize a extração e seleção de features manualmente. SVR e CART são exemplos de modelos clássicos.

 - Redes Neurais (Modelos Conexionistas): A característica definidora é a sua estrutura em camadas de neurônios interconectados. Modelos com uma profundidade maior de camadas ocultas (como LSTM ou CNN) são frequentemente chamados de modelos de Deep Learning (DL).

#### Uma RN é um modelo de ML? Por quê?
Sim, uma Rede Neural é um modelo de Machine Learning (Aprendizado de Máquina).

 - Definição de Machine Learning: Machine Learning é um campo que reúne métodos baseados em álgebra, probabilidades, estatística e otimização, capazes de extrair conhecimento dos dados com o objetivo de reconhecimento e detecção de padrões e previsão de resultados futuros.

 - RN se Enquadra: As Redes Neurais se encaixam perfeitamente nessa definição porque elas:

    - Utilizam otimização matemática (algoritmos como backpropagation e Adam).
    - Extraem conhecimento e padrões dos dados (através do ajuste de seus pesos).
    - Têm como objetivo a generalização e a previsão de resultados futuros.

Em suma, Machine Learning é o campo abrangente, e as Redes Neurais (incluindo o Deep Learning) são uma subárea, classe ou técnica poderosa dentro desse campo.[citados em varios artigos]