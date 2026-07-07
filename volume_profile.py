from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ==========================================================
# DATACLASS
# ==========================================================

@dataclass
class VolumeProfile:
    """
    Risultato Volume Profile.
    """
    prices: np.ndarray
    volumes: np.ndarray
    poc: float
    vah: float
    val: float
    total_volume: float
    bins: int

    def as_dataframe(self) -> pd.DataFrame:
        """
        Restituisce il profilo come DataFrame.
        """
        return pd.DataFrame(
            {
                "price": self.prices,
                "volume": self.volumes
            }
        )


# ==========================================================
# PRICE RANGE
# ==========================================================

def calculate_price_range(df: pd.DataFrame):
    """
    Determina il range massimo/minimo del periodo analizzato.
    """
    return (
        df["Low"].min(),
        df["High"].max()
    )


# ==========================================================
# PRICE BINS
# ==========================================================

def create_price_bins(low: float, high: float, bins: int):
    return np.linspace(low, high, bins + 1)


# ==========================================================
# VOLUME DISTRIBUTION (LOGICA TRADINGVIEW AVANZATA)
# ==========================================================

def distribute_volume(df: pd.DataFrame, price_bins: np.ndarray) -> np.ndarray:
    """
    Distribuisce il volume proporzionalmente sull'intervallo High-Low di ciascuna candela.
    Risolve il problema della concentrazione su un unico livello (Typical Price).
    """
    profile = np.zeros(len(price_bins) - 1)
    
    # Estrazione array per massimizzare le performance del ciclo
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    volumes = df["Volume"].values

    for h, l, c, vol in zip(highs, lows, closes, volumes):
        if vol <= 0:
            continue
            
        # Gestione candele a range zero o micro-range (evita divisioni per zero)
        if h <= l:
            idx = np.searchsorted(price_bins, c) - 1
            idx = max(0, min(idx, len(profile) - 1))
            profile[idx] += vol
            continue

        # Individua i bin che si sovrappongono al range [l, h] della candela
        # Usiamo un approccio geometrico di intersezione segmenti
        for i in range(len(profile)):
            b_low = price_bins[i]
            b_high = price_bins[i + 1]

            # Calcolo dell'intersezione tra il bin corrente [b_low, b_high] e la candela [l, h]
            overlap_low = max(l, b_low)
            overlap_high = min(h, b_high)

            if overlap_high > overlap_low:
                # Proporzione del range della candela che cade in questo bin
                fraction = (overlap_high - overlap_low) / (h - l)
                profile[i] += vol * fraction

    return profile


# ==========================================================
# POC
# ==========================================================

def calculate_poc(volumes: np.ndarray, prices: np.ndarray) -> float:
    """
    Point Of Control: livello di prezzo con il massimo volume scambiato.
    """
    index = np.argmax(volumes)
    return prices[index]


# ==========================================================
# VALUE AREA
# ==========================================================

def calculate_value_area(
    prices: np.ndarray,
    volumes: np.ndarray,
    percentage: float = 0.70
) -> tuple[float, float]:
    """
    Calcola VAH e VAL tramite espansione bidirezionale simmetrica dal POC
    fino a coprire la percentuale target (es. 70%) del volume totale.
    """
    total_volume = volumes.sum()
    if total_volume <= 0:
        return prices[0], prices[-1]

    target = total_volume * percentage
    poc_index = np.argmax(volumes)

    included = {poc_index}
    current_volume = volumes[poc_index]

    left = poc_index - 1
    right = poc_index + 1

    while current_volume < target:
        left_volume = volumes[left] if left >= 0 else -1
        right_volume = volumes[right] if right < len(volumes) else -1

        # Se entrambi i lati sono esauriti, interrompi
        if left_volume == -1 and right_volume == -1:
            break

        if left_volume >= right_volume:
            if left >= 0:
                included.add(left)
                current_volume += volumes[left]
                left -= 1
        else:
            if right < len(volumes):
                included.add(right)
                current_volume += volumes[right]
                right += 1

    low_idx = min(included)
    high_idx = max(included)

    return prices[low_idx], prices[high_idx]


# ==========================================================
# API PUBBLICA
# ==========================================================

def calculate_volume_profile(
    df: pd.DataFrame,
    bins: int = 50,
    value_area: float = 0.70
) -> Optional[VolumeProfile]:
    """
    Calcola il Volume Profile completo.
    """
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
    
    # I prezzi dei bin corrispondono al punto medio di ciascun intervallo
    prices = (edges[:-1] + edges[1:]) / 2

    poc = calculate_poc(volumes, prices)
    val, vah = calculate_value_area(prices, volumes, value_area)

    return VolumeProfile(
        prices=prices,
        volumes=volumes,
        poc=float(poc),
        vah=float(vah),
        val=float(val),
        total_volume=float(volumes.sum()),
        bins=bins
    )


# ==========================================================
# UTILITY
# ==========================================================

def distance_from_poc(price: float, poc: float) -> float:
    """
    Distanza percentuale del prezzo corrente dal POC.
    """
    if poc == 0:
        return 0.0
    return abs(price - poc) / poc * 100


def price_near_poc(price: float, poc: float, tolerance: float = 2.0) -> bool:
    """
    True se il prezzo si trova all'interno della tolleranza percentuale dal POC.
    """
    return distance_from_poc(price, poc) <= tolerance


# ==========================================================
# TEST LOGICA COMPLETA
# ==========================================================
if __name__ == "__main__":
    from data import load_ticker
    from zigzag import detect_macro_swing

    df = load_ticker("AAPL", 500)
    swing = detect_macro_swing(df)

    if swing is None:
        print("Nessuno swing trovato.")
    else:
        # Il profilo viene calcolato SOLO sulle barre dello swing
        swing_df = swing.data(df)
        profile = calculate_volume_profile(swing_df)

        print("\n" + "="*40)
        print("VOLUME PROFILE OUTPUT (LOGICA REALE)")
        print("="*40)
        print(f"POC: {profile.poc:.2f}")
        print(f"VAL: {profile.val:.2f}")
        print(f"VAH: {profile.vah:.2f}")
        print(f"Volume Totale Area: {profile.total_volume:,.0f}")
        print("="*40)