from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as o

# ==========================================================
# MAPPING TICKER TRADINGVIEW -> YAHOO FINANCE
# ==========================================================
# Yahoo Finance non riconosce la sintassi dei Future di TradingView (es. ETH1!).
# Questo dizionario converte i simboli prima del download.
TICKER_MAPPING = {
    "ETH1!": "ETH-USD",  # Spot Crypto (consigliato su Yahoo per storici continui e volumi puliti)
    "BTC1!": "BTC-USD",  # Spot Crypto
    "ES1!": "ES=F",      # Future E-mini S&P 500
    "NQ1!": "NQ=F",      # Future E-mini Nasdaq 100
    "YM1!": "YM=F",      # Future Dow Jones
    "RTY1!": "RTY=F",    # Future Russell 2000
    "GC1!": "GC=F",      # Future Gold
    "CL1!": "CL=F",      # Future Crude Oil
}

def get_yahoo_ticker(ticker: str) -> str:
    """Restituisce il ticker corretto per Yahoo Finance."""
    return TICKER_MAPPING.get(ticker.upper().strip(), ticker)


# ==========================================================
# STRUTTURE DATI (DATACLASSES)
# ==========================================================

@dataclass
class VolumeProfile:
    """Risultato del calcolo del Volume Profile."""
    prices: np.ndarray
    volumes: np.ndarray
    poc: float
    vah: float
    val: float
    total_volume: float
    bins: int

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({"price": self.prices, "volume": self.volumes})


@dataclass
class PivotPoint:
    """Rappresenta un punto di pivot (Massimo o Minimo)."""
    index: any
    position: int
    price: float
    is_high: bool


@dataclass
class Swing:
    """
    Rappresenta un movimento Macro Swing individuato sul grafico.
    Contiene gli indici temporali e le posizioni assolute (iloc).
    """
    start_index: any
    end_index: any
    start_pos: int
    end_pos: int
    start_price: float
    end_price: float
    is_up: bool

    def data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estrazione classica limitata strettamente tra i due pivot."""
        return df.iloc[self.start_pos : self.end_pos + 1].copy()

    def data_to_current(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        LOGICA TRADINGVIEW: Estrae i dati dal pivot iniziale fino all'ultima
        barra disponibile sul grafico (oggi), includendo lo storico recente nel VP.
        """
        return df.iloc[self.start_pos :].copy()


# ==========================================================
# LOGICA DI CALCOLO VOLUME PROFILE
# ==========================================================

def calculate_price_range(df: pd.DataFrame) -> tuple[float, float]:
    return float(df["Low"].min()), float(df["High"].max())


def create_price_bins(low: float, high: float, bins: int) -> np.ndarray:
    return np.linspace(low, high, bins + 1)


def distribute_volume(df: pd.DataFrame, price_bins: np.ndarray) -> np.ndarray:
    """Distribuisce il volume geometricamente lungo il range High-Low di ogni candela."""
    profile = np.zeros(len(price_bins) - 1)
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    volumes = df["Volume"].values

    for h, l, c, vol in zip(highs, lows, closes, volumes):
        if vol <= 0:
            continue
        if h <= l:
            idx = np.searchsorted(price_bins, c) - 1
            idx = max(0, min(idx, len(profile) - 1))
            profile[idx] += vol
            continue

        for i in range(len(profile)):
            b_low = price_bins[i]
            b_high = price_bins[i + 1]
            overlap_low = max(l, b_low)
            overlap_high = min(h, b_high)

            if overlap_high > overlap_low:
                fraction = (overlap_high - overlap_low) / (h - l)
                profile[i] += vol * fraction
    return profile


def calculate_poc(volumes: np.ndarray, prices: np.ndarray) -> float:
    return float(prices[np.argmax(volumes)])


def calculate_value_area(
    prices: np.ndarray, volumes: np.ndarray, percentage: float = 0.70
) -> tuple[float, float]:
    total_volume = volumes.sum()
    if total_volume <= 0:
        return float(prices[0]), float(prices[-1])

    target = total_volume * percentage
    poc_index = int(np.argmax(volumes))
    included = {poc_index}
    current_volume = float(volumes[poc_index])
    left, right = poc_index - 1, poc_index + 1

    while current_volume < target:
        left_volume = volumes[left] if left >= 0 else -1
        right_volume = volumes[right] if right < len(volumes) else -1

        if left_volume == -1 and right_volume == -1:
            break

        if left_volume >= right_volume and left_volume != -1:
            included.add(left)
            current_volume += volumes[left]
            left -= 1
        elif right_volume != -1:
            included.add(right)
            current_volume += volumes[right]
            right += 1
        else:
            break

    low_idx, high_idx = min(included), max(included)
    return float(prices[low_idx]), float(prices[high_idx])


def calculate_volume_profile(
    df: pd.DataFrame, bins: int = 50, value_area: float = 0.70
) -> Optional[VolumeProfile]:
    if df is None or df.empty:
        return None
    required = ["High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in required):
        return None

    low, high = calculate_price_range(df)
    if low == high:
        return None

    edges = create_price_bins(low, high, bins)
    volumes = distribute_volume(df, edges)
    prices = (edges[:-1] + edges[1:]) / 2
    poc = calculate_poc(volumes, prices)
    val, vah = calculate_value_area(prices, volumes, value_area)

    return VolumeProfile(
        prices=prices, volumes=volumes, poc=poc, vah=vah, val=val, total_volume=float(volumes.sum()), bins=bins
    )


# ==========================================================
# DETECT MACRO SWING SIMULATO (ZIGZAG AUTONOMO)
# ==========================================================

def detect_macro_swing(df: pd.DataFrame, window: int = 20, atr_period: int = 14, min_atr_ratio: float = 2.0) -> Optional[Swing]:
    """
    Versione standalone semplificata di detect_macro_swing per garantire il funzionamento autonomo.
    Identifica l'ultimo macro movimento basato sui massimi e minimi di periodo.
    """
    if len(df) < window * 2:
        return None
        
    # Identificazione pivot grezzi di massima/minima di periodo nelle ultime battute
    high_peaks = []
    low_peaks = []
    
    for i in range(window, len(df) - window):
        sub_high = df['High'].iloc[i-window:i+window+1].max()
        sub_low = df['Low'].iloc[i-window:i+window+1].min()
        
        if df['High'].iloc[i] == sub_high:
            high_peaks.append((df.index[i], i, df['High'].iloc[i]))
        if df['Low'].iloc[i] == sub_low:
            low_peaks.append((df.index[i], i, df['Low'].iloc[i]))

    if not high_peaks or not low_peaks:
        # Fallback statico protettivo basato sul range massimo se non trova pivot geometrici puliti
        p1_idx, p2_idx = int(len(df) * 0.6), int(len(df) * 0.9)
        return Swing(
            start_index=df.index[p1_idx], end_index=df.index[p2_idx],
            start_pos=p1_idx, end_pos=p2_idx,
            start_price=df['Low'].iloc[p1_idx], end_price=df['High'].iloc[p2_idx],
            is_up=True
        )

    # Prendi gli ultimi due strutturati per creare lo swing di test
    p1 = low_peaks[-1] if low_peaks[-1][1] < high_peaks[-1][1] else high_peaks[-1]
    p2 = high_peaks[-1] if p1[1] < high_peaks[-1][1] else low_peaks[-1]
    
    if p1[1] >= p2[1]:
        if len(low_peaks) > 1: p1 = low_peaks[-2]
        else: return None

    return Swing(
        start_index=p1[0], end_index=p2[0],
        start_pos=p1[1], end_pos=p2[1],
        start_price=p1[2], end_price=p2[2],
        is_up=(p2[2] > p1[2])
    )


# ==========================================================
# CARICAMENTO DATI DA YAHOO FINANCE
# ==========================================================

def load_ticker(ticker_name: str, lookback: int) -> pd.DataFrame:
    """Scarica dati storici puliti applicando il corretto mappaggio dei ticker."""
    yahoo_symbol = get_yahoo_ticker(ticker_name)
    periodo = "1mo" if lookback <= 30 else "3mo" if lookback <= 90 else "1y" if lookback <= 365 else "max"
    
    df = yf.download(yahoo_symbol, period=periodo, interval="1d")
    if df.empty:
        raise ValueError(f"Nessun dato restituito da Yahoo Finance per il simbolo: {yahoo_symbol}")
    
    # Pulizia colonne multi-index se presenti nel nuovo formato yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    return df.tail(lookback)


# ==========================================================
# GRAFICA GENERATIVA (PLOTLY)
# ==========================================================

def create_chart(df: pd.DataFrame, swing: Swing, profile: VolumeProfile, ema_period: int = 21) -> o.Figure:
    fig = o.Figure()
    
    # Candlestick principale
    fig.add_trace(o.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="Prezzo", opacity=0.4
    ))
    
    # Ema di controllo
    df['EMA'] = df['Close'].ewm(span=ema_period, adjust=False).mean()
    fig.add_trace(o.Scatter(x=df.index, y=df['EMA'], line=dict(color='orange', width=1.5), name=f"EMA {ema_period}"))

    if profile and swing:
        # Estendiamo visivamente le linee del profilo da start_index fino a oggi (df.index[-1])
        x_start = swing.start_index
        x_end = df.index[-1]
        
        # Linea POC
        fig.add_trace(o.Scatter(
            x=[x_start, x_end], y=[profile.poc, profile.poc],
            line=dict(color='red', width=2.5, dash='solid'), name=f"POC: {profile.poc:.2f}"
        ))
        # Linea VAH
        fig.add_trace(o.Scatter(
            x=[x_start, x_end], y=[profile.vah, profile.vah],
            line=dict(color='cyan', width=1.5, dash='dash'), name=f"VAH: {profile.vah:.2f}"
        ))
        # Linea VAL
        fig.add_trace(o.Scatter(
            x=[x_start, x_end], y=[profile.val, profile.val],
            line=dict(color='cyan', width=1.5, dash='dash'), name=f"VAL: {profile.val:.2f}"
        ))

    fig.update_layout(xaxis_rangeslider_visible=False, template="plotly_dark", height=600)
    return fig


# ==========================================================
# CORE APP / STREAMLIT RENDER (APP.PY)
# ==========================================================

def main():
    st.set_page_config(layout="wide")
    st.title("Macro Swing & Anchored Volume Profile (TV-Style)")

    # Sidebar controlli parametri
    st.sidebar.header("Parametri Configurazione")
    selected = st.sidebar.text_input("Inserisci Simbolo (es. ETH1!, BTC1!, ES1!, AAPL)", value="ETH1!")
    lookback = st.sidebar.slider("Lookback Candele", min_value=50, max_value=1000, value=300)
    
    swing_window = st.sidebar.slider("Finestra Swing (Pivot)", 5, 50, 20)
    atr_period = st.sidebar.slider("Periodo ATR", 5, 30, 14)
    min_atr_ratio = st.sidebar.slider("Min ATR Ratio", 0.5, 5.0, 2.0, step=0.1)
    ema_period = st.sidebar.slider("Periodo EMA", 10, 200, 21)

    try:
        # Caricamento e computazione
        df = load_ticker(selected, lookback)
        
        swing = detect_macro_swing(
            df,
            window=swing_window,
            atr_period=atr_period,
            min_atr_ratio=min_atr_ratio
        )

        if swing is None:
            st.warning("Macro Swing non disponibile con i parametri attuali.")
        else:
            # ----------------=======================================
            # BLOCCO TEMPORANEO DI SEVERO DEBUG (STREAMLIT NATIVE)
            # ----------------=======================================
            st.write("### 🔍 SEZIONE DI DEBUG POSIZIONAMENTO & COPERTURA VOLUMI")
            
            c1, c2 = st.columns(2)
            with c1:
                st.metric(label="Inizio Swing (Data/Index)", value=str(swing.start_index))
                st.metric(label="Fine Swing (Data/Index)", value=str(swing.end_index))
                st.metric(label="Ultima Candela Totale DF", value=str(df.index[-1]))
            with c2:
                st.metric(label="Start Position Absolute (iloc)", value=int(swing.start_pos))
                st.metric(label="End Position Absolute (iloc)", value=int(swing.end_pos))
                st.metric(label="Ultima Posizione DF Abs (len-1)", value=int(len(df) - 1))

            # --- CORREZIONE LOGICA ANCHORED PROFILE FINO A OGGI ---
            swing_df = swing.data_to_current(df)
            # ------------------------------------------------------

            st.write("#### 📊 Statistiche Quantitative di Controllo")
            st.info(f"""
            * **Numero totale di barre disponibili nel DataFrame:** {len(df)}
            * **Numero di barre effettivamente fornite al motore VP:** {len(swing_df)} *(Inizio Swing ➔ Oggi)*
            * **Indice temporale dell'ultima candela calcolata dal VP:** `{swing_df.index[-1]}`
            """)
            
            # Calcolo Volume Profile sulle barre espanse ed estese fino ad oggi
            profile = calculate_volume_profile(swing_df)
            
            # Generazione e stampa del grafico Plotly
            fig = create_chart(df, swing, profile, ema_period=ema_period)
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Errore durante l'esecuzione del programma: {e}")

if __name__ == "__main__":
    main()