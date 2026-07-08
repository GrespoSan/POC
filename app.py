from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go

# ==========================================================
# MAPPING TICKER TRADINGVIEW -> YAHOO FINANCE
# ==========================================================
TICKER_MAPPING = {
    "ETH1!": "ETH-USD",  
    "BTC1!": "BTC-USD",  
    "ES1!": "ES=F",      
    "NQ1!": "NQ=F",      
    "GC1!": "GC=F",      
    "CL1!": "CL=F",      
}

def get_yahoo_ticker(ticker: str) -> str:
    return TICKER_MAPPING.get(ticker.upper().strip(), ticker)


# ==========================================================
# DATACLASSES RIGIDAMENTE SEPARATE
# ==========================================================

@dataclass
class VolumeProfile:
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
class Swing:
    """
    Rappresenta ESCLUSIVAMENTE la struttura geometrica dello swing rilevato.
    È totalmente indipendente dal Volume Profile.
    """
    start_index: any
    end_index: any
    start_pos: int
    end_pos: int
    start_price: float
    end_price: float
    is_up: bool

    def data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rappresenta lo swing puro (inizio -> fine swing). Usato per grafici/statistiche dello swing."""
        return df.iloc[self.start_pos : self.end_pos + 1].copy()

    def profile_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Dati da usare per il Volume Profile Anchored (inizio swing -> OGGI).
        Non modifica l'essenza dello swing originale.
        """
        return df.iloc[self.start_pos :].copy()


# ==========================================================
# MOTORE DI CALCOLO VOLUME PROFILE
# ==========================================================

def calculate_price_range(df: pd.DataFrame) -> tuple[float, float]:
    return float(df["Low"].min()), float(df["High"].max())

def create_price_bins(low: float, high: float, bins: int) -> np.ndarray:
    return np.linspace(low, high, bins + 1)

def distribute_volume(df: pd.DataFrame, price_bins: np.ndarray) -> np.ndarray:
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

def calculate_value_area(prices: np.ndarray, volumes: np.ndarray, percentage: float = 0.70) -> tuple[float, float]:
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

def calculate_volume_profile(df: pd.DataFrame, bins: int = 50, value_area: float = 0.70) -> Optional[VolumeProfile]:
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
# ALGORITMO RILEVAMENTO MACRO SWING
# ==========================================================
def detect_macro_swing(df: pd.DataFrame, window: int = 20, atr_period: int = 14, min_atr_ratio: float = 2.0) -> Optional[Swing]:
    """
    Rileva i macro swing basandosi esclusivamente sulle candele del DataFrame primario.
    """
    if len(df) < window * 2:
        return None
        
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
        # Fallback deterministico
        p1_idx, p2_idx = int(len(df) * 0.6), int(len(df) * 0.9)
        return Swing(
            start_index=df.index[p1_idx], end_index=df.index[p2_idx],
            start_pos=p1_idx, end_pos=p2_idx,
            start_price=df['Low'].iloc[p1_idx], end_price=df['High'].iloc[p2_idx],
            is_up=True
        )

    p1 = low_peaks[-1] if low_peaks[-1][1] < high_peaks[-1][1] else high_peaks[-1]
    p2 = high_peaks[-1] if p1[1] < high_peaks[-1][1] else low_peaks[-1]
    
    if p1[1] >= p2[1] and len(low_peaks) > 1:
        p1 = low_peaks[-2]

    return Swing(
        start_index=p1[0], end_index=p2[0],
        start_pos=p1[1], end_pos=p2[1],
        start_price=p1[2], end_price=p2[2],
        is_up=(p2[2] > p1[2])
    )


# ==========================================================
# CARICAMENTO DATI
# ==========================================================
def load_ticker(ticker_name: str, lookback: int) -> pd.DataFrame:
    yahoo_symbol = get_yahoo_ticker(ticker_name)
    periodo = "1mo" if lookback <= 30 else "3mo" if lookback <= 90 else "1y" if lookback <= 365 else "max"
    
    df = yf.download(yahoo_symbol, period=periodo, interval="1d")
    if df.empty:
        raise ValueError(f"Nessun dato per il simbolo: {yahoo_symbol}")
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    return df.tail(lookback)


# ==========================================================
# GRAFICA PLOTLY (UTILIZZA I SEGMENTI SEPARATI CORRETTI)
# ==========================================================
def create_chart(df: pd.DataFrame, swing: Swing, profile: VolumeProfile) -> go.Figure:
    fig = go.Figure()
    
    # Grafico Candlestick completo
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="Prezzo", opacity=0.3
    ))
    
    # GRAFICO SWING: Segmento reale (inizio -> fine) richiesto per la visualizzazione grafica dello swing
    swing_slice = df.loc[swing.start_index : swing.end_index]
    fig.add_trace(go.Scatter(
        x=[swing.start_index, swing.end_index],
        y=[swing.start_price, swing.end_price],
        line=dict(color='magenta', width=3),
        mode='lines+markers',
        name="Macro Swing Reale"
    ))

    # GRAFICO VOLUME PROFILE: Proiettato dall'ancora iniziale (start_index) fino a OGGI (df.index[-1])
    if profile:
        x_start = swing.start_index
        x_end = df.index[-1]
        
        fig.add_trace(go.Scatter(
            x=[x_start, x_end], y=[profile.poc, profile.poc],
            line=dict(color='red', width=2.5), name=f"Anchored POC: {profile.poc:.2f}"
        ))
        fig.add_trace(go.Scatter(
            x=[x_start, x_end], y=[profile.vah, profile.vah],
            line=dict(color='cyan', width=1.5, dash='dash'), name=f"Anchored VAH: {profile.vah:.2f}"
        ))
        fig.add_trace(go.Scatter(
            x=[x_start, x_end], y=[profile.val, profile.val],
            line=dict(color='cyan', width=1.5, dash='dash'), name=f"Anchored VAL: {profile.val:.2f}"
        ))

    fig.update_layout(xaxis_rangeslider_visible=False, template="plotly_dark", height=600)
    return fig


# ==========================================================
# FLUSSO APPLICATIVO INTERFACCIA
# ==========================================================
def main():
    st.set_page_config(layout="wide")
    st.title("Disaccoppiamento Strutturale: Swing vs Volume Profile")

    selected = st.sidebar.text_input("Simbolo", value="ETH1!")
    lookback = st.sidebar.slider("Lookback", 50, 1000, 300)
    swing_window = st.sidebar.slider("Finestra Swing", 5, 50, 20)

    try:
        df = load_ticker(selected, lookback)
        
        # 1. Lo Swing viene calcolato e rimane immutabile
        swing = detect_macro_swing(df, window=swing_window)

        if swing is None:
            st.warning("Nessun macro swing rilevato.")
        else:
            # 2. Generiamo il profilo usando ESCLUSIVAMENTE la nuova funzione dedicata
            profile_df = swing.profile_data(df)
            profile = calculate_volume_profile(profile_df)

            # Debug di convalida strutturale
            st.write("### 🔍 VALIDAZIONE SEPARAZIONE FLUSSI")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Lunghezza Swing Reale (Barre)", len(swing.data(df)))
                st.write(f"Segmento: `{swing.start_index}` ➔ `{swing.end_index}`")
            with c2:
                st.metric("Lunghezza Calcolo VP (Barre)", len(profile_df))
                st.write(f"Segmento: `{swing.start_index}` ➔ `{df.index[-1]}`")
            with c3:
                st.metric("Ultima Barra del Dataframe", str(df.index[-1]))
                st.write(f"Ultima Barra usata nel VP: `{profile_df.index[-1]}`")

            # 3. Il grafico riceve gli oggetti puliti e disegna lo swing reale ma il VP ancorato esteso
            fig = create_chart(df, swing, profile)
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Errore: {e}")

if __name__ == "__main__":
    main()