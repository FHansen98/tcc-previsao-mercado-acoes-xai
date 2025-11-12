Usar a Biblioteca Python yfinance (Solução Robusta e Recomendada)
Esta é a forma mais fiável e gratuita para automatizar e obter grandes volumes de dados históricos, contornando a interface web:

Instale o Python (se ainda não o tiver) e instale as bibliotecas necessárias:

Bash

pip install yfinance pandas
Execute o Código: Use o seguinte código para descarregar os dados diretamente e guardá-los num arquivo CSV no seu computador:

Python

import yfinance as yf
import pandas as pd

# Símbolo do Bovespa
ticker = "^BVSP" 

# Define as datas desejadas (exemplo: desde 1 de Janeiro de 2000 até hoje)
dados = yf.download(ticker, start="2000-01-01")

# Guardar os dados num ficheiro CSV
dados.to_csv("BVSP_dados_historicos.csv")

print(f"Dados do {ticker} descarregados e guardados em BVSP_dados_historicos.csv")
Resultado: Um novo ficheiro chamado BVSP_dados_historicos.csv será criado na mesma pasta onde executar o código, contendo toda a informação histórica (Open, High, Low, Close, Adj Close, Volume) para o período especificado.