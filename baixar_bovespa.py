import yfinance as yf
import pandas as pd

# Símbolo do Bovespa
ticker = "^BVSP" 

# Define as datas desejadas (exemplo: desde 1 de Janeiro de 2000 até hoje)
dados = yf.download(ticker, start="2000-01-01")

# Guardar os dados num ficheiro CSV
dados.to_csv("BVSP_dados_historicos.csv")

print(f"Dados do {ticker} descarregados e guardados em BVSP_dados_historicos.csv")
