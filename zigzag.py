from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import math


# ==========================================================
# DATACLASSES
# ==========================================================

@dataclass
class Pivot:
    """
    Rappresentazione di un singolo punto di inversione (Pivot).
    """
    index: pd.Timestamp
    position: int
    price: float
    kind: str          # "high" oppure "low"


@dataclass
class Swing:
    """
    Rappresentazione di un movimento direzionale (Macro Swing).
    """
    start: Pivot
    end: Pivot
    direction: str     # "UP" / "DOWN" / "SIDE"
    move_pct: float
    bars: int
    score: float = 0.0
    atr_ratio: float = 0.0

    @property
    def start_pos(self) -> int:
        return self.start.position

    @property
    def end_pos(self) -> int:
        return self.end.position

    @property
    def start_price(self) -> float:
        return self.start.price

    @property
    def end_price(self) -> float:
        return self.end.price

    @property
    def start_index(self) -> pd.Timestamp:
        return self.start.index

    @property
    def end_index(self) -> pd.Timestamp:
        return self.end.index

    def data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Estrae la porzione di DataFrame inclusa tra l'inizio e la fine dello swing.
        """
        return df.iloc[self.start.position:self.end.position + 1].copy()

    @property
    def length(self) -> int:
        return self.end.position - self.start.position + 1

    @property
    def is_rising(self) -> bool:
        return self.direction == "UP"

    @property
    def is_falling(self) -> bool:
        return self.direction == "DOWN"
        
    @property
    def is_sideways(self) -> bool:
        return self.direction == "SIDE"

    def summary(self) -> dict:
        return {
            "direction": self.direction,
            "start": self.start_index,
            "end": self.end_index,
            "start_price": self.start_price,
            "end_price": self.end_price,
            "move_pct": round(self.move_pct, 2),
            "bars": self.bars,
            "atr_ratio": round(self.atr_ratio, 2),
            "score": round(self.score, 2)
        }


# ==========================================================
# MATHEMATICAL UTILITIES
# ==========================================================

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calcola l'Average True Range (ATR) classico per la misurazione della volatilità.
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ],
        axis=1
    ).max(axis=1)

    return tr.rolling(period).mean()


def movement_percent(a: float, b: float) -> float:
    if a == 0:
        return 0.0
    return (abs(b - a) / a) * 100


def swing_direction(start_price: float, end_price: float, threshold: float = 0.1) -> str:
    if start_price == 0:
        return "SIDE"
        
    change = ((end_price - start_price) / start_price) * 100

    if change > threshold:
        return "UP"
    if change < -threshold:
        return "DOWN"
    return "SIDE"


def clamp(value, low, high):
    return max(low, min(value, high))


def normalize(value, min_value, max_value):
    # CORREZIONE: Se i valori coincidono (es. un solo swing), il punteggio deve essere massimo (1.0)
    if max_value == min_value:
        return 1.0
    return (value - min_value) / (max_value - min_value)


def exponential_recency(position: int, total: int, alpha: float = 3.0) -> float:
    """
    Assegna un moltiplicatore di peso (0.0 - 1.0) basato sulla vicinanza temporale.
    """
    if total <= 1:
        return 1.0
    x = position / (total - 1)
    return (math.exp(alpha * x) - 1) / (math.exp(alpha) - 1)


# ==========================================================
# FILTERS
# ==========================================================

def remove_duplicate_pivots(pivots: List[Pivot]) -> List[Pivot]:
    """
    Elimina i pivot consecutivi appartenenti allo stesso tipo, isolando l'estremo reale.
    """
    if len(pivots) < 2:
        return pivots

    filtered = []
    current = pivots[0]

    for nxt in pivots[1:]:
        if nxt.kind != current.kind:
            filtered.append(current)
            current = nxt
            continue

        if current.kind == "high":
            if nxt.price > current.price:
                current = nxt
        else:
            if nxt.price < current.price:
                current = nxt

    filtered.append(current)
    return filtered


# ==========================================================
# PIVOT DETECTOR
# ==========================================================

class PivotDetector:
    """
    Esamina lo storico prezzi alla ricerca di massimi e minimi locali significativi.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        window: int = 5,
        atr_period: int = 14,
        min_atr_ratio: float = 1.5
    ):
        self.df = df.copy()
        self.window = window
        self.atr = calculate_atr(self.df, atr_period)
        self.min_atr_ratio = min_atr_ratio
        
        self._highs = self.df["High"].values
        self._lows = self.df["Low"].values
        self._atr_values = self.atr.values

    def detect(self) -> List[Pivot]:
        pivots = []

        # Scansione della finestra mobile escludendo i margini esterni
        for i in range(self.window, len(self.df) - self.window):
            if self._is_pivot_high(i):
                pivots.append(
                    Pivot(
                        index=self.df.index[i],
                        position=i,
                        price=float(self._highs[i]),
                        kind="high"
                    )
                )
            elif self._is_pivot_low(i):
                pivots.append(
                    Pivot(
                        index=self.df.index[i],
                        position=i,
                        price=float(self._lows[i]),
                        kind="low"
                    )
                )

        # Applica prima la rimozione dei duplicati grezzi dello stesso tipo
        pivots = remove_duplicate_pivots(pivots)
        # Filtra le oscillazioni insignificanti (rumore monetario) rispetto all'ATR
        pivots = self._filter_noise(pivots)
        return pivots

    def _is_pivot_high(self, pos: int) -> bool:
        value = self._highs[pos]
        left = self._highs[pos - self.window : pos]
        right = self._highs[pos + 1 : pos + self.window + 1]
        return value > left.max() and value > right.max()

    def _is_pivot_low(self, pos: int) -> bool:
        value = self._lows[pos]
        left = self._lows[pos - self.window : pos]
        right = self._lows[pos + 1 : pos + self.window + 1]
        return value < left.min() and value < right.min()

    def _filter_noise(self, pivots: List[Pivot]) -> List[Pivot]:
        if len(pivots) < 2:
            return pivots

        filtered = [pivots[0]]

        for pivot in pivots[1:]:
            last = filtered[-1]
            atr = self._atr_values[pivot.position]

            if pd.isna(atr) or atr <= 0:
                continue

            # CORREZIONE: Applica il filtro della distanza ATR solo se sono pivot alternati.
            # Se per rumore strutturale sono dello stesso tipo, mantieni l'alternanza logica.
            if pivot.kind == last.kind:
                if pivot.kind == "high" and pivot.price > last.price:
                    filtered[-1] = pivot
                elif pivot.kind == "low" and pivot.price < last.price:
                    filtered[-1] = pivot
                continue

            distance = abs(pivot.price - last.price)
            ratio = distance / atr

            dynamic_ratio = self.min_atr_ratio
            if pivot.kind != last.kind:
                dynamic_ratio *= 0.8  # Agevola le transizioni strutturali pulite

            if ratio >= dynamic_ratio:
                filtered.append(pivot)

        return remove_duplicate_pivots(filtered)


# ==========================================================
# SWING BUILDER
# ==========================================================

class SwingBuilder:
    """
    Genera segmenti d'onda vettoriali unendo i nodi pivot validati.
    """
    def __init__(self, pivots: List[Pivot]):
        self.pivots = pivots

    def build(self) -> List[Swing]:
        swings = []

        if len(self.pivots) < 2:
            return swings

        for i in range(len(self.pivots) - 1):
            start = self.pivots[i]
            end = self.pivots[i + 1]

            if end.position <= start.position:
                continue

            move = movement_percent(start.price, end.price)
            bars = end.position - start.position

            swing = Swing(
                start=start,
                end=end,
                direction=swing_direction(start.price, end.price),
                move_pct=move,
                bars=bars
            )
            swings.append(swing)

        return swings


# ==========================================================
# MACRO SWING DETECTOR
# ==========================================================

class MacroSwingDetector:
    """
    Valuta e classifica la rilevanza dei movimenti per estrarre l'onda primaria.
    """
    def __init__(self, df: pd.DataFrame, swings: List[Swing], atr: pd.Series):
        self.df = df
        self.swings = swings
        self.atr = atr
        self._atr_values = atr.values

    def detect(self) -> Optional[Swing]:
        if len(self.swings) == 0:
            return None

        self._compute_atr_ratio()
        self._score()

        return max(self.swings, key=lambda s: s.score)

    def _compute_atr_ratio(self):
        for swing in self.swings:
            atr = self._atr_values[swing.end.position]

            if pd.isna(atr) or atr == 0:
                swing.atr_ratio = 0.0
                continue

            distance = abs(swing.end.price - swing.start.price)
            swing.atr_ratio = distance / atr

    def _score(self):
        if len(self.swings) == 0:
            return

        move_values = [s.move_pct for s in self.swings]
        bar_values = [s.bars for s in self.swings]
        atr_values = [s.atr_ratio for s in self.swings]

        min_move, max_move = min(move_values), max(move_values)
        min_bar, max_bar = min(bar_values), max(bar_values)
        min_atr, max_atr = min(atr_values), max(atr_values)

        total = len(self.swings)

        for i, swing in enumerate(self.swings):
            move_score = normalize(swing.move_pct, min_move, max_move)
            bar_score = normalize(swing.bars, min_bar, max_bar)
            atr_score = normalize(swing.atr_ratio, min_atr, max_atr)
            recency = exponential_recency(i, total)

            # Sistema di ponderazione bilanciato: 35% Ampiezza, 25% Volatilità, 10% Durata, 30% Recenza
            swing.score = (
                move_score * 35.0 +
                atr_score * 25.0 +
                bar_score * 10.0 +
                recency * 30.0
            )


# ==========================================================
# API PUBBLICA
# ==========================================================

def detect_macro_swing(
    df: pd.DataFrame,
    window: int = 5,
    atr_period: int = 14,
    min_atr_ratio: float = 1.5,
    min_score: float = 20.0  # Abbassato leggermente il filtro per evitare scarti iper-conservativi
) -> Optional[Swing]:
    """
    Punto di accesso principale per identificare il miglior Macro Swing strutturale del grafico.
    """
    if df is None or df.empty:
        return None

    required = ["High", "Low", "Close"]
    if not all(col in df.columns for col in required):
        return None

    if len(df) < window * 3:
        return None

    atr = calculate_atr(df, atr_period)

    detector = PivotDetector(
        df=df,
        window=window,
        atr_period=atr_period,
        min_atr_ratio=min_atr_ratio
    )
    pivots = detector.detect()

    if len(pivots) < 2:
        return None

    builder = SwingBuilder(pivots)
    swings = builder.build()

    if len(swings) == 0:
        return None

    macro = MacroSwingDetector(df, swings, atr).detect()

    if macro is None:
        return None

    if macro.score < min_score:
        return None

    return macro


# ==========================================================
# TEST EXECUTABLE
# ==========================================================

if __name__ == "__main__":
    try:
        from data import load_ticker
        df_test = load_ticker("AAPL", 500)
        
        swing_result = detect_macro_swing(df_test)

        if swing_result is None:
            print("Nessun macro swing strutturale rilevato con i parametri correnti.")
        else:
            print("\n" + "=" * 50)
            print(" SYSTEM LOG: ZIGZAG MACRO SWING DETECTED")
            print("=" * 50)
            import pprint
            pprint.pprint(swing_result.summary())
            print("=" * 50)
            
    except ImportError:
        print("Sintassi OK. Modulo di caricamento 'data' non presente localmente.")