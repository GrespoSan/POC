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

def calculate_price_range(
    df: pd.DataFrame
):
    """
    Determina il range massimo/minimo
    del periodo analizzato.
    """

    return (
        df["Low"].min(),
        df["High"].max()
    )


# ==========================================================
# PRICE BINS
# ==========================================================

def create_price_bins(
    low: float,
    high: float,
    bins: int
):

    return np.linspace(
        low,
        high,
        bins + 1
    )


# ==========================================================
# VOLUME DISTRIBUTION
# ==========================================================

def distribute_volume(
    df: pd.DataFrame,
    price_bins: np.ndarray
):

    """
    Distribuisce il volume sulle fasce prezzo.

    Metodo:
    - usa il prezzo medio candela
    - assegna tutto il volume
      al relativo bucket
    """

    typical_price = (
        df["High"]
        +
        df["Low"]
        +
        df["Close"]
    ) / 3


    volume = df["Volume"]


    indices = np.digitize(
        typical_price,
        price_bins
    )


    profile = np.zeros(
        len(price_bins)-1
    )


    for idx, vol in zip(
        indices,
        volume
    ):

        if 0 < idx <= len(profile):

            profile[idx-1] += vol


    return profile



# ==========================================================
# POC
# ==========================================================

def calculate_poc(
    volumes,
    prices
):

    """
    Point Of Control.

    Prezzo con massimo volume.
    """

    index = np.argmax(
        volumes
    )

    return prices[index]



# ==========================================================
# VALUE AREA
# ==========================================================

def calculate_value_area(
    prices,
    volumes,
    percentage: float = 0.70
):

    """
    Calcola VAH e VAL.

    Metodo:
    espansione dal POC
    fino al raggiungimento del 70%
    del volume totale.
    """

    total_volume = volumes.sum()

    target = (
        total_volume *
        percentage
    )


    poc_index = np.argmax(
        volumes
    )


    included = {
        poc_index
    }


    current_volume = (
        volumes[poc_index]
    )


    left = poc_index - 1
    right = poc_index + 1


    while current_volume < target:


        left_volume = (
            volumes[left]
            if left >= 0
            else -1
        )


        right_volume = (
            volumes[right]
            if right < len(volumes)
            else -1
        )


        if left_volume >= right_volume:

            if left >= 0:
                included.add(left)

                current_volume += (
                    volumes[left]
                )

                left -= 1

            else:
                break


        else:

            if right < len(volumes):

                included.add(right)

                current_volume += (
                    volumes[right]
                )

                right += 1

            else:
                break


    low = min(included)
    high = max(included)


    return (
        prices[low],
        prices[high]
    )



# ==========================================================
# API PUBBLICA
# ==========================================================

def calculate_volume_profile(
    df: pd.DataFrame,
    bins: int = 50,
    value_area: float = 0.70
) -> Optional[VolumeProfile]:

    """
    Calcola il Volume Profile.

    Parameters
    ----------
    df:
        DataFrame OHLCV relativo
        al macro swing

    bins:
        Numero livelli prezzo

    value_area:
        Percentuale volume Value Area

    Returns
    -------
    VolumeProfile
    """


    if df is None:
        return None


    if df.empty:
        return None


    required = [
        "High",
        "Low",
        "Close",
        "Volume"
    ]


    if not all(
        c in df.columns
        for c in required
    ):
        return None



    low, high = calculate_price_range(
        df
    )


    if low == high:
        return None



    edges = create_price_bins(
        low,
        high,
        bins
    )


    volumes = distribute_volume(
        df,
        edges
    )


    prices = (
        edges[:-1]
        +
        edges[1:]
    ) / 2



    poc = calculate_poc(
        volumes,
        prices
    )


    val, vah = calculate_value_area(
        prices,
        volumes,
        value_area
    )


    return VolumeProfile(

        prices=prices,

        volumes=volumes,

        poc=float(poc),

        vah=float(vah),

        val=float(val),

        total_volume=float(
            volumes.sum()
        ),

        bins=bins

    )



# ==========================================================
# UTILITY
# ==========================================================

def distance_from_poc(
    price: float,
    poc: float
):

    """
    Distanza percentuale dal POC.
    """

    if poc == 0:
        return 0

    return abs(
        price-poc
    ) / poc * 100



def price_near_poc(
    price: float,
    poc: float,
    tolerance: float = 2.0
):

    """
    True se il prezzo è entro
    la tolleranza dal POC.
    """

    return (
        distance_from_poc(
            price,
            poc
        )
        <= tolerance
    )



# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    from data import load_ticker
    from zigzag import detect_macro_swing


    df = load_ticker(
        "AAPL",
        500
    )


    swing = detect_macro_swing(
        df
    )


    if swing is None:

        print(
            "Nessuno swing trovato"
        )


    else:

        swing_df = swing.data(
            df
        )


        profile = calculate_volume_profile(
            swing_df
        )


        print(
            profile
        )


        print()

        print(
            "POC:",
            round(profile.poc,2)
        )

        print(
            "VAL:",
            round(profile.val,2)
        )

        print(
            "VAH:",
            round(profile.vah,2)
        )
