from __future__ import annotations


from dataclasses import dataclass
from typing import List, Optional


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



    def to_dict(self):

        return {

            "Ticker": self.ticker,

            "Price": round(
                self.price,
                2
            ),

            "POC": round(
                self.poc,
                2
            ),

            "VAL": round(
                self.val,
                2
            ),

            "VAH": round(
                self.vah,
                2
            ),

            "POC Distance %": round(
                self.poc_distance,
                2
            ),

            "Trend": self.direction,

            "Swing %": round(
                self.swing_move,
                2
            ),

            "Score": round(
                self.swing_score,
                2
            ),

            "Bars": self.swing_bars,

            "ATR Ratio": round(
                self.atr_ratio,
                2
            )

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
    Analizza un singolo titolo.
    """


    if df is None or df.empty:

        return None



    swing = detect_macro_swing(
        df,
        window=swing_window,
        atr_period=atr_period,
        min_atr_ratio=min_atr_ratio
    )


    if swing is None:

        return None



    swing_df = swing.data(
        df
    )


    profile = calculate_volume_profile(
        swing_df
    )


    if profile is None:

        return None



    price = float(
        df["Close"].iloc[-1]
    )


    poc_distance = distance_from_poc(
        price,
        profile.poc
    )


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
    loader,
    swing_window: int = 5,
    atr_period: int = 14,
    min_atr_ratio: float = 1.5,
    poc_tolerance: float = 3.0,
    progress_callback=None
) -> pd.DataFrame:

    """
    Analizza una lista ticker.

    Parameters
    ----------
    tickers:
        Lista simboli

    loader:
        Funzione che carica OHLCV

        esempio:

        load_ticker(ticker, period)

    progress_callback:
        funzione opzionale per Streamlit


    Returns
    -------
    DataFrame risultati
    """


    results = []


    total = len(tickers)



    for i, ticker in enumerate(tickers):


        try:

            df = loader(
                ticker,
                500
            )


            result = analyze_ticker(

                ticker,

                df,

                swing_window,

                atr_period,

                min_atr_ratio,

                poc_tolerance

            )


            if result:

                results.append(
                    result.to_dict()
                )


        except Exception:

            # uno stock problematico
            # non deve fermare lo screening

            continue



        if progress_callback:

            progress_callback(
                (i+1)/total
            )



    if len(results)==0:

        return pd.DataFrame()



    output = pd.DataFrame(
        results
    )


    # Ranking:
    # prima vicino al POC
    # poi swing score

    output = output.sort_values(
        [
            "POC Distance %",
            "Score"
        ],
        ascending=[
            True,
            False
        ]
    )


    output.reset_index(
        drop=True,
        inplace=True
    )


    return output



# ==========================================================
# UTILITIES
# ==========================================================


def load_tickers(
    filename="tickers.txt"
) -> List[str]:

    """
    Carica lista ticker da file.
    """

    with open(
        filename,
        "r"
    ) as f:

        tickers = [

            line.strip().upper()

            for line in f

            if line.strip()

        ]


    return tickers



# ==========================================================
# TEST
# ==========================================================


if __name__ == "__main__":


    from data import load_ticker



    tickers = [

        "AAPL",
        "MSFT",
        "NVDA",
        "TSLA"

    ]



    df = run_screening(

        tickers,

        load_ticker

    )


    print()

    print(
        df
    )

