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

def add_context_lines(fig, profile):
    """
    Aggiunge le linee chiave del Volume Profile con stili stabili e compatibili.
    """
    # POC - Rosso continuo, evidenziato con Arial Black
    fig.add_hline(
        y=profile.poc,
        line_width=2,
        line_color="#EF5350",
        annotation_text="POC",
        annotation_position="top left",
        annotation_font=dict(color="#EF5350", size=11, family="Arial Black")
    )

    # VAH - Tratteggiato scuro standard
    fig.add_hline(
        y=profile.vah,
        line_dash="dash",
        line_width=1.5,
        line_color="#78909C",
        annotation_text="VAH",
        annotation_position="top left",
        annotation_font=dict(color="#78909C", size=10, family="Arial")
    )

    # VAL - Tratteggiato scuro standard
    fig.add_hline(
        y=profile.val,
        line_dash="dash",
        line_width=1.5,
        line_color="#78909C",
        annotation_text="VAL",
        annotation_position="bottom left",
        annotation_font=dict(color="#78909C", size=10, family="Arial")
    )


# ==========================================================
# VOLUME PROFILE TRACE (X2 ASSE INDIPENDENTE)
# ==========================================================

def add_volume_profile(fig, profile):
    """
    Disegna Volume Profile su un asse X secondario (xaxis2) sovrapposto sulla destra.
    """
    max_volume = profile.volumes.max()
    widths = profile.volumes / max_volume

    fig.add_trace(
        go.Bar(
            x=widths,
            y=profile.prices,
            orientation="h",
            name="Volume Profile",
            opacity=0.30,
            marker=dict(color="#455A64"),
            xaxis="x2",
            hovertemplate="Prezzo: %{y:.2f}<br>Volume Relativo: %{x:.2f}<extra></extra>"
        )
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
    Lo swing viene tracciato seguendo la curva dei prezzi reali tra i due pivot.
    """
    fig = go.Figure()

    # 1. Candlestick completo (Mantiene asse temporale coerente e totale)
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

    # 2. EMA su intero tracciato
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

    # 3. Macro Swing mappato sull'andamento continuo dei prezzi reali
    if swing:
        # Estrazione della fetta di dati racchiusa nello swing per mappare l'andamento continuo
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

    # 4. Livelli del Volume Profile (POC, VAH, VAL)
    if profile:
        add_context_lines(fig, profile)

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

    # 6. Tracciamento Volume Profile Orizzontale a destra
    if profile and show_volume_profile:
        add_volume_profile(fig, profile)

    # Cuscinetto protettivo a destra (Evita sovrapposizioni grafiche tra barre e volume profile)
    end_date_buffered = df.index[-1] + timedelta(days=5)

    # 7. Layout finale ad alta risoluzione (Fissato a 950px, autosize abilitato)
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
            r=80,
            t=50,
            b=50
        ),
        
        xaxis=dict(
            domain=[0, 0.78],
            rangeslider_visible=False,
            title="Data",
            range=[df.index[0], end_date_buffered]  # Forza la visualizzazione fino a oggi + buffer
        ),
        
        xaxis2=dict(
            domain=[0.78, 1],
            overlaying="y",
            side="top",
            showgrid=False,
            title="Volume Relativo",
            ticks="",
            showticklabels=False
        ),
        
        yaxis=dict(
            fixedrange=False,
            side="left",
            title="Prezzo"
        )
    )

    return fig