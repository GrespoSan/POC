import pandas as pd
import streamlit as st
import yfinance as yf


@st.cache_data(show_spinner=False)
def download_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Scarica lo storico di un ticker da Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Simbolo del titolo.
    period : str
        Periodo richiesto (es. '1y', '2y', '5y').

    Returns
    -------
    DataFrame
        DataFrame OHLCV pulito.
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

    if df.empty:
        return pd.DataFrame()

    # Se le colonne sono MultiIndex (può accadere con yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Manteniamo solo le colonne utili
    expected = ["Open", "High", "Low", "Close", "Volume"]

    df = df[[c for c in expected if c in df.columns]]

    # Conversione numerica
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Elimina righe con valori mancanti
    df = df.dropna()

    # Volume valido
    df = df[df["Volume"] > 0]

    # Ordine cronologico
    df.sort_index(inplace=True)

    return df


def get_last_price(df: pd.DataFrame) -> float:
    """Restituisce l'ultimo prezzo di chiusura."""

    if df.empty:
        return None

    return float(df["Close"].iloc[-1])


def get_last_volume(df: pd.DataFrame) -> int:
    """Restituisce l'ultimo volume."""

    if df.empty:
        return None

    return int(df["Volume"].iloc[-1])


def get_period_from_lookback(lookback: int) -> str:
    """
    Converte i giorni richiesti nel periodo Yahoo.

    Parameters
    ----------
    lookback : int

    Returns
    -------
    str
    """

    if lookback <= 250:
        return "1y"

    if lookback <= 500:
        return "2y"

    if lookback <= 750:
        return "3y"

    if lookback <= 1000:
        return "5y"

    return "10y"


def load_ticker(ticker: str, lookback: int) -> pd.DataFrame:
    """
    Scarica e restituisce solamente gli ultimi N giorni.
    """

    period = get_period_from_lookback(lookback)

    df = download_data(
        ticker,
        period=period
    )

    if df.empty:
        return df

    return df.tail(lookback).copy()