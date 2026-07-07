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
    Il POC attraversa il grafico dall'inizio dello swing fino ad oggi.
    VAH e VAL rimangono racchiusi nell'area di trading dello swing.
    """
    # POC dinamico (da inizio swing fino all'ultima data disponibile)
    fig.add_shape(
        type="line",
        x0=swing.start_index,
        x1=last_date,
        y0=profile.poc,
        y1=profile.poc,
        line=dict(color="#00897B", width=3),  # Verde acqua scuro stile TV
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
        font=dict(color="#00897B", size=11, family="Arial Black")
    )

    # VAH Linea segmentata (confinata nel range dello swing)
    fig.add_shape(
        type="line",
        x0=swing.start_index,
        x1=swing.end_index,
        y0=profile.vah,
        y1=profile.vah,
        line=dict(color="#78909C", width=1.5, dash="dash"),
    )

    # VAL Linea segmentata (confinata nel range dello swing)
    fig.add_shape(
        type="line",
        x0=swing.start_index,
        x1=swing.end_index,
        y0=profile.val,
        y1=profile.val,
        line=dict(color="#78909C", width=1.5, dash="dash"),
    )


# ==========================================================
# VOLUME PROFILE TRACE (STILE TRADINGVIEW ANCORATO)
# ==========================================================

def add_volume_profile(fig, profile, swing):
    """
    Disegna il Volume Profile ancorato esattamente alla data di inizio del macro swing,
    proiettando le barre orizzontalmente verso destra sovrapposte al prezzo storico.
    """
    max_volume = profile.volumes.max()
    widths = profile.volumes / max_volume

    # Calcoliamo lo spessore delle barre (altezza in prezzo del singolo bucket bin)
    bin_width = profile.prices[1] - profile.prices[0] if len(profile.prices) > 1 else 1.0

    # Base di ancoraggio iniziale (corrisponde all'inizio dello swing rilevato)
    start_date = swing.start_index
    
    # Creiamo le shape rettangolari proiettando i giorni in avanti in base all'intensità volumetrica
    for price, width in zip(profile.prices, widths):
        if width <= 0:
            continue
            
        # Estensione orizzontale proporzionale (es. ampiezza massima di 12 giorni di contrattazione)
        x_end = start_date + timedelta(days=int(width * 12))
        
        fig.add_shape(
            type="rect",
            x0=start_date,
            x1=x_end,
            y0=price - (bin_width / 2),
            y1=price + (bin_width / 2),
            fillcolor="rgba(0, 180, 180, 0.20)",  # Colore azzurro TradingView semitrasparente
            line=dict(width=0),
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
    Il Volume Profile e i livelli chiave si integrano allineandosi alla timeline corretta.
    """
    fig = go.Figure()

    # 1. Candlestick completo (Usa il dataframe completo fino a oggi)
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

    # 3. Struttura del Macro Swing lungo la curva dei prezzi reali
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

    # 6. Volume Profile Ancorato all'inizio dello Swing
    if profile and swing and show_volume_profile:
        add_volume_profile(fig, profile, swing)

    # Cuscinetto temporale di sicurezza a destra oltre l'ultima candela
    end_date_buffered = df.index[-1] + timedelta(days=15)

    # 7. Layout ad alte prestazioni con asse dei prezzi a destra (Professional Trading Style)
    fig.update_layout(
        title=dict(
            text="POC Macro Swing & Market Structure Analysis (TradingView Style)",
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
            r=60,
            t=60,
            b=50
        ),
        
        xaxis=dict(
            rangeslider_visible=False,
            title="Data",
            range=[df.index[0], end_date_buffered]
        ),
        
        yaxis=dict(
            fixedrange=False,
            side="right",  # Scala dei prezzi a destra come l'immagine di riferimento
            title="Prezzo"
        )
    )

    return fig