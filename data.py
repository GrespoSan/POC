from __future__ import annotations

import pandas as pd
import streamlit as st
import yfinance as yf


@st.cache_data(show_spinner=False)
def download_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Scarica lo storico di un ticker da Yahoo Finance e lo normalizza.
    """
    try:
        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False
        )
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # Se le colonne sono MultiIndex (comportamento predefinito di yfinance),
    # appiattiamo il primo livello per isolare Open, High, Low, Close, Volume
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Rimuoviamo l'eventuale fuso orario dall'indice per evitare incompatibilità con Plotly
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Manteniamo esclusivamente le colonne utili alla struttura OHLCV dello screener
    expected = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in expected if c in df.columns]].copy()

    # Conversione forzata a tipo numerico per evitare anomalie con tipi object o stringhe
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Elimina righe contenenti valori mancanti (NaN)
    df = df.dropna()

    # Mantiene solo le sessioni con volumi scambiati attivi (> 0)
    df = df[df["Volume"] > 0]

    # Garantisce il perfetto ordine cronologico crescente delle date
    df.sort_index(inplace=True)

    return df


def get_last_price(df: pd.DataFrame) -> float | None:
    """Restituisce l'ultimo prezzo di chiusura disponibile."""
    if df.empty:
        return None
    return float(df["Close"].iloc[-1])


def get_last_volume(df: pd.DataFrame) -> int | None:
    """Restituisce l'ultimo volume registrato a mercato."""
    if df.empty:
        return None
    return int(df["Volume"].iloc[-1])


def get_period_from_lookback(lookback: int) -> str:
    """
    Converte il lookback richiesto in barre di borsa nel rispettivo 
    intervallo temporale macro accettato da Yahoo Finance.
    """
    # I giorni richiesti si riferiscono alle barre di borsa aperte (circa 252 all'anno)
    if lookback <= 200:
        return "1y"
    if lookback <= 450:
        return "2y"
    if lookback <= 700:
        return "3y"
    if lookback <= 1100:
        return "5y"
    return "10y"


def load_ticker(ticker: str, lookback: int) -> pd.DataFrame:
    """
    Scarica lo storico necessario e restituisce esattamente la finestra temporale 
    richiesta dall'utente per l'analisi del macro swing.
    """
    # Determina l'intervallo di download ottimale
    period = get_period_from_lookback(lookback)

    df = download_data(
        ticker,
        period=period
    )

    if df.empty:
        return df

    # Estrae la coda esatta corrispondente al numero di candele (lookback) desiderate
    return df.tail(lookback).copy()