"""Coleta S&P500 desde 2000-01-01 para validacao-lstm-final.

Uso:
    python src/coleta.py

Saída:
    data/sp500_clean.parquet (6.288 dias, 2000-2024)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'validacao-bolsa' / 'src'))

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).parent.parent / 'data'
DATA_DIR.mkdir(exist_ok=True, parents=True)

IND = 'sp500'
START_DATE = '2000-01-01'
END_DATE = '2024-12-31'

def main():
    print(f"Coletando {IND} desde {START_DATE} até {END_DATE}...")
    df = yf.download('^GSPC', start=START_DATE, end=END_DATE, auto_adjust=False)
    df = df.reset_index()
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    df = df.sort_values('Date').reset_index(drop=True)

    # Imputar volume zero
    df['Volume'] = df['Volume'].replace(0, pd.NA)
    df['Volume'] = df['Volume'].interpolate(limit_direction='both').fillna(df['Volume'].median())
    df['Volume_imputed'] = df['Volume'].isna().astype(int)

    # Criar coluna Price (usamos Close)
    df['Price'] = df['Close']

    # Salvar (CSV para evitar dependência pyarrow)
    output_path = DATA_DIR / f'{IND}_clean.csv'
    df.to_csv(output_path, index=False)

    print(f"\n=== Coleta concluída ===")
    print(f"Linhas: {len(df)}")
    print(f"Período: {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"Salvo em: {output_path}")

if __name__ == '__main__':
    main()
