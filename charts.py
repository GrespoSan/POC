"""
charts.py
==========

Rendering grafico professionale stile TradingView.
Il Volume Profile viene renderizzato a destra della candela corrente sul grafico principale,
e il POC si estende dall'inizio dello swing fino all'ultima seduta di mercato.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta


# ==========================================================
# EMA
# ==========================================================

def calculate_ema(df: pd.DataFrame, period: int = 200) -> pd.Series:
    """
    Calcola EMA sul dataset complessivo.
    """
    return df["Close"].ewm(span=period, adjust=False).mean()


# ==========================================================
# FIBONACCI
# ==========================================================

def fibonacci_levels(swing) -> dict:
    """
    Calcola livelli Fibonacci del macro swing.
    """
    high = max(swing.start_price, swing.end_price)
    low = min(swing.start_price, swing.end_price)
    diff = high - low

    levels = {
        "0.0": high,
        "23.6": high - diff * 0.236,
        "38.2": high - diff * 0.382,
        "50.0": high - diff * 0.5,
        "61.8": high - diff * 0.618,
        "78.6": high - diff * 0.786,
        "100": low
    }
    return levels


# ==========================================================
# HORIZONTAL LINES STYLING
# ==========================================================

def add_context_lines(fig, profile, swing, last_date):
    """
    Aggiunge le linee chiave del Volume Profile stile TradingView.
    Il POC viene disegnato come una shape lineare che parte dallo swing e arriva a oggi.
    VAH e VAL seguono lo stesso principio per pulizia grafica.
    """
    # POC dinamico (da inizio swing a oggi)
    fig.add_shape(
        type="line",
        x0=swing.start_index,
        x1=last_date,
        y0=profile.poc,
        y1=profile.poc,
        line=dict(color="#EF5350", width=3),
    )
    
    # Etichetta testo per il POC
    fig.add_annotation(
        x=swing.start_index,
        y=profile.poc,
        text="POC",
        xref="x",
        yref="y",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(color="#EF5350", size=11, family="Arial Black")
    )

    # VAH Linea segmentata
    fig.add_shape(
        type="line",
        x0=swing.start_index,
        x1=last_date,
        y0=profile.vah,
        y1=profile.vah,
        line=dict(color="#78909C", width=1.5, dash="dash"),
    )

    # VAL Linea segmentata
    fig.add_shape(
        type="line",
        x0=swing.start_index,
        x1=last_date,
        y0=profile.val,
        y1=profile.val,
        line=dict(color="#78909C", width=1.5, dash="dash"),
    )


# ==========================================================
# VOLUME PROFILE TRACE (STILE TRADINGVIEW)
# ==========================================================

def add_volume_profile(fig, profile, df):
    """
    Disegna il Volume Profile sul lato destro vicino alla candela corrente,
    proiettando le barre orizzontalmente nell'area del margine destro salvaguardato.
    """
    max_volume = profile.volumes.max()
    widths = profile.volumes / max_volume

    # Calcoliamo lo spessore delle barre (altezza in prezzo del singolo bucket bin)
    bin_width = profile.prices[1] - profile.prices[0] if len(profile.prices) > 1 else 1.0

    # Base di posizionamento sull'asse delle date (subito dopo l'ultima candela disponibile)
    last_date = df.index[-1]
    
    # Creiamo la proiezione delle barre aggiungendo giorni proporzionali al volume relativo
    # 15 giorni rappresenta la massima estensione visiva sul grafico a destra della candela corrente
    for price, width in zip(profile.prices, widths):
        if width <= 0:
            continue
            
        x_end = last_date + timedelta(days=int(width * 15))
        
        fig.add_shape(
            type="rect",
            x0=last_date,
            x1=x_end,
            y0=price - (bin_width / 2),
            y1=price + (bin_width / 2),
            fillcolor="rgba(0, 180, 180, 0.35)",
            line=dict(width=0), # Rimuove il bordo per un look minimale e pulito
        )


# ==========================================================
# MAIN CHART FUNCTION
# ==========================================================

def create_chart(
    df: pd.DataFrame,
    swing=None,
    profile=None,
    show_volume_profile: bool = True,
    show_fibonacci: bool = True,
    ema_period: int = 200
) -> go.Figure:
    """
    Crea un grafico completo mantenendo visibile lo storico fino alla candela corrente.
    Il Volume Profile e i livelli chiave si integrano organicamente sul lato destro del trend.
    """
    fig = go.Figure()

    # 1. Candlestick completo (Usa il dataframe completo)
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color="#26A69A", 
            decreasing_line_color="#EF5350"
        )
    )

    # 2. EMA su tutto lo storico
    ema = calculate_ema(df, ema_period)
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=ema,
            name=f"EMA {ema_period}",
            mode="lines",
            line=dict(color="#2196F3", width=1.5)
        )
    )

    # 3. Struttura del Macro Swing lungo il prezzo reale
    if swing:
        swing_slice = df.loc[swing.start_index : swing.end_index]
        fig.add_trace(
            go.Scatter(
                x=swing_slice.index,
                y=swing_slice["Close"],
                mode="lines",
                name="Macro Swing Structure",
                line=dict(color="#FF9800", width=3.5),
            )
        )

    # 4. Livelli e POC stile TradingView (Da inizio swing alla candela corrente)
    if profile and swing:
        add_context_lines(fig, profile, swing, df.index[-1])

    # 5. Livelli Fibonacci
    if swing and show_fibonacci:
        fibs = fibonacci_levels(swing)
        for name, price in fibs.items():
            fig.add_hline(
                y=price,
                line_dash="dot",
                line_width=1,
                line_color="#B0BEC5",
                annotation_text=f"Fib {name}",
                annotation_position="bottom right",
                annotation_font=dict(color="#90A4AE", size=9, family="Arial")
            )

    # 6. Volume Profile sul margine destro del grafico principale
    if profile and show_volume_profile:
        add_volume_profile(fig, profile, df)

    # Estendiamo lo spazio a destra dell'asse X per alloggiare visivamente il Volume Profile (18 giorni totali)
    end_date_buffered = df.index[-1] + timedelta(days=18)

    # 7. Layout finale ad alta risoluzione a griglia unificata
    fig.update_layout(
        title=dict(
            text="POC Macro Swing & Market Structure Analysis",
            font=dict(size=18, family="Arial Black")
        ),
        height=950,
        autosize=True,
        template="plotly_white",
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        
        margin=dict(
            l=50,
            r=50,
            t=60,
            b=50
        ),
        
        xaxis=dict(
            rangeslider_visible=False,
            title="Data",
            range=[df.index[0], end_date_buffered]  # Mantiene visibile l'intero dataset + spazio volume profile
        ),
        
        yaxis=dict(
            fixedrange=False,
            side="left",
            title="Prezzo"
        )
    )

    return fig