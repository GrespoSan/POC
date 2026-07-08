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

    profile_df = swing.profile_data(df)
    profile = calculate_volume_profile(profile_df)

    if profile is None:
        return None

    price = float(df["Close"].iloc[-1])
    poc_distance = distance_from_poc(price, profile.poc)

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
    lookback: int = 500,  # <-- AGGIUNTO E COORDINATO CON APP.PY
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
            # Passa correttamente il lookback al loader personalizzato
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
    output = output.sort_values(
        by=["POC Distance %", "Score"],
        ascending=[True, False]
    )
    output.reset_index(drop=True, inplace=True)
    return output