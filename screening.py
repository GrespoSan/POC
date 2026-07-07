from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Callable
import pandas as pd

from zigzag import detect_macro_swing
from volume_profile import (
    calculate_volume_profile,
    distance_from_poc
)


# ==========================================================
# RESULT DATACLASS
# ==========================================================

@dataclass
class ScreeningResult:
    """
    Risultato analisi singolo ticker.
    """
    ticker: str
    price: float
    poc: float
    val: float
    vah: float
    poc_distance: float
    direction: str
    swing_move: float
    swing_score: float
    swing_bars: int
    atr_ratio: float

    def to_dict(self) -> dict:
        return {
            "Ticker": self.ticker,
            "Price": round(self.price, 2),
            "POC": round(self.poc, 2),
            "VAL": round(self.val, 2),
            "VAH": round(self.vah, 2),
            "POC Distance %": round(self.poc_distance, 2),
            "Trend": self.direction,
            "Swing %": round(self.swing_move, 2),
            "Score": round(self.swing_score, 2),
            "Bars": self.swing_bars,
            "ATR Ratio": round(self.atr_ratio, 2)
        }


# ==========================================================
# SINGLE TICKER ANALYSIS
# ==========================================================

def analyze_ticker(
    ticker: str,
    df: pd.DataFrame,
    swing_window: int = 5,
    atr_period: int = 14,
    min_atr_ratio: float = 1.5,
    poc_tolerance: float = 3.0
) -> Optional[ScreeningResult]:
    """
    Analizza un singolo titolo calcolando macro swing e volume profile.
    """
    if df is None or df.empty or len(df) < max(swing_window, atr_period):
        return None

    swing = detect_macro_swing(
        df,
        window=swing_window,
        atr_period=atr_period,
        min_atr_ratio=min_atr_ratio
    )

    if swing is None:
        return None

    # Estrae la fetta di dati racchiusa nel macro swing
    swing_df = swing.data(df)
    
    # Calcola il volume profile geometrico distribuito sul range High-Low
    profile = calculate_volume_profile(swing_df)

    if profile is None:
        return None

    # Ultimo prezzo di chiusura disponibile sul mercato
    price = float(df["Close"].iloc[-1])

    # Distanza percentuale simmetrica rispetto al POC di riferimento dello swing
    poc_distance = distance_from_poc(price, profile.poc)

    # Filtro di tolleranza impostato dall'utente
    if poc_distance > poc_tolerance:
        return None

    return ScreeningResult(
        ticker=ticker,
        price=price,
        poc=profile.poc,
        val=profile.val,
        vah=profile.vah,
        poc_distance=poc_distance,
        direction=swing.direction,
        swing_move=swing.move_pct,
        swing_score=swing.score,
        swing_bars=swing.bars,
        atr_ratio=swing.atr_ratio
    )


# ==========================================================
# BATCH SCREENER
# ==========================================================

def run_screening(
    tickers: List[str],
    loader: Callable[[str, int], pd.DataFrame],
    lookback: int = 500,
    swing_window: int = 5,
    atr_period: int = 14,
    min_atr_ratio: float = 1.5,
    poc_tolerance: float = 3.0,
    progress_callback: Optional[Callable[[float], None]] = None
) -> pd.DataFrame:
    """
    Analizza una lista di ticker applicando i filtri di prossimità volumetrica al POC.
    """
    results = []
    total = len(tickers)

    if total == 0:
        return pd.DataFrame()

    for i, ticker in enumerate(tickers):
        try:
            # Carica lo storico applicando il lookback dinamico dell'interfaccia
            df = loader(ticker, lookback)

            result = analyze_ticker(
                ticker=ticker,
                df=df,
                swing_window=swing_window,
                atr_period=atr_period,
                min_atr_ratio=min_atr_ratio,
                poc_tolerance=poc_tolerance
            )

            if result:
                results.append(result.to_dict())

        except Exception:
            continue

        if progress_callback:
            progress_callback((i + 1) / total)

    if not results:
        return pd.DataFrame()

    output = pd.DataFrame(results)

    # Ranking: ordina prima per la massima vicinanza al POC (Ascending)
    # poi per la forza dello swing (Descending)
    output = output.sort_values(
        by=["POC Distance %", "Score"],
        ascending=[True, False]
    )

    output.reset_index(drop=True, inplace=True)
    return output


# ==========================================================
# UTILITIES
# ==========================================================

def load_tickers(filename: str = "tickers.txt") -> List[str]:
    """
    Carica una lista pulita di ticker da file di testo locale.
    """
    try:
        with open(filename, "r") as f:
            return [
                line.strip().upper()
                for line in f
                if line.strip()
            ]
    except FileNotFoundError:
        return []


# ==========================================================
# TEST EXECUTABLE
# ==========================================================

if __name__ == "__main__":
    from data import load_ticker

    test_tickers = ["AAPL", "MSFT", "NVDA", "TSLA"]
    
    df_results = run_screening(
        tickers=test_tickers,
        loader=load_ticker,
        lookback=500
    )

    print("\n--- RISULTATI SCREENING TEST ---")
    print(df_results)