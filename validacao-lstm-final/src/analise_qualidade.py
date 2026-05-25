"""Análise de qualidade dos dados S&P500 (2000-2024).

Gera relatório de estatísticas e insights para modelagem.
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data'
RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

IND = 'sp500'

def main():
    print(f"Carregando dados {IND}...")
    df = pd.read_csv(DATA_DIR / f'{IND}_clean.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    print(f"Período: {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"Linhas: {len(df)}")

    # Retornos
    df['ret_1d'] = df['Close'].pct_change()
    df['logret_1d'] = np.log(df['Close'] / df['Close'].shift(1))

    # Estatísticas
    ret_clean = df['ret_1d'].dropna()
    logret_clean = df['logret_1d'].dropna()

    stats = {
        'linhas_totais': int(len(df)),
        'periodo_inicio': str(df['Date'].min().date()),
        'periodo_fim': str(df['Date'].max().date()),
        'missings_ohlc': int(df[['Open', 'High', 'Low', 'Close']].isna().sum().sum()),
        'volume_zero': int((df['Volume'] == 0).sum()),
        'duplicatas_data': int(df['Date'].duplicated().sum()),
        'retorno_acumulado': float((1 + ret_clean).prod() - 1),
        'vol_anualizada': float(ret_clean.std() * np.sqrt(252)),
        'retorno_medio_diario': float(ret_clean.mean()),
        'desvio_padrao_diario': float(ret_clean.std()),
        'skewness': float(ret_clean.skew()),
        'curtose_excesso': float(ret_clean.kurtosis()),
        'pior_dia': float(ret_clean.min()),
        'pior_dia_data': str(df.loc[ret_clean.idxmin(), 'Date'].date()),
        'melhor_dia': float(ret_clean.max()),
        'melhor_dia_data': str(df.loc[ret_clean.idxmax(), 'Date'].date()),
        'close_min': float(df['Close'].min()),
        'close_max': float(df['Close'].max()),
    }

    print("\n=== Estatísticas ===")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    # Salvar relatório
    import json
    relatorio_path = RESULTS_DIR / f'fase0_qualidade_{IND}_2000_2024.json'
    with open(relatorio_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\nRelatório salvo em: {relatorio_path}")

    # Insights
    print("\n=== Insights para modelagem ===")
    print(f"- Curtose excesso = {stats['curtose_excesso']:.1f} (>11) → caudas pesadas")
    print(f"- Skewness = {stats['skewness']:.2f} (negativo) → quedas grandes mais comuns")
    print(f"- Volatilidade anual = {stats['vol_anualizada']:.1%}")
    print(f"- Retorno acumulado 2000-2024 = {stats['retorno_acumulado']:.1%}")

if __name__ == '__main__':
    main()
