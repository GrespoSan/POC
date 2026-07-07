"""
charts.py
==========

Rendering grafico professionale con asse X secondario sovrapposto per il Volume Profile.
Focus incentrato sul Macro Swing selezionato per evitare distorsioni visive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


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
    Aggiunge le linee chiave del Volume Profile con stili professionali:
    - POC: Linea continua Rossa spessa.
    - VAH/VAL: Linee tratteggiate Grigio/Nere.
    """
    # POC (Point of Control) - Rosso continuo ben visibile
    fig.add_hline(
        y=profile.poc,
        line_width=2,
        line_color="#EF5350",
        annotation_text="POC",
        annotation_position="top left",
        annotation_font=dict(color="#EF5350", size=11, bold=True)
    )

    # VAH (Value Area High) - Tratteggiato scuro
    fig.add_hline(
        y=profile.vah,
        line_dash="dash",
        line_width=1.5,
        line_color="#78909C",
        annotation_text="VAH",
        annotation_position="top left",
        annotation_font=dict(color="#78909C", size=10)
    )

    # VAL (Value Area Low) - Tratteggiato scuro
    fig.add_hline(
        y=profile.val,
        line_dash="dash",
        line_width=1.5,
        line_color="#78909C",
        annotation_text="VAL",
        annotation_position="bottom left",
        annotation_font=dict(color="#78909C", size=10)
    )


# ==========================================================
# VOLUME PROFILE TRACE (X2 ASSE INDIPENDENTE)
# ==========================================================

def add_volume_profile(fig, profile):
    """
    Disegna Volume Profile su un asse X secondario (xaxis2) sovrapposto sulla destra.
    Evita il bug dello schiacciamento temporale.
    """
    max_volume = profile.volumes.max()
    widths = profile.volumes / max_volume

    fig.add_trace(
        go.Bar(
            x=widths,
            y=profile.prices,
            orientation="h",
            name="Volume Profile",
            opacity=0.35,
            marker=dict(color="#455A64"),
            xaxis="x2",  # Mappato esplicitamente sul secondo asse X
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
    Crea un grafico con focus incentrato sullo swing (+50 barre precedenti).
    Spazio diviso: 78% Prezzo/Candele, 22% Volume Profile verticale sulla destra.
    """
    
    # Ritaglio del frame per il focus visivo richiesto
    if swing:
        start_idx = max(0, swing.start_pos - 50)
        view_df = df.iloc[start_idx : swing.end_pos + 1].copy()
    else:
        view_df = df.copy()

    fig = go.Figure()

    # 1. Candlestick (Asse X standard)
    fig.add_trace(
        go.Candlestick(
            x=view_df.index,
            open=view_df["Open"],
            high=view_df["High"],
            low=view_df["Low"],
            close=view_df["Close"],
            name="Price",
            increasing_line_color="#26A69A", 
            decreasing_line_color="#EF5350"
        )
    )

    # 2. EMA (Calcolata sul df globale per accuratezza, visualizzata sul df tagliato)
    ema_global = calculate_ema(df, ema_period)
    fig.add_trace(
        go.Scatter(
            x=view_df.index,
            y=ema_global.loc[view_df.index],
            name=f"EMA {ema_period}",
            mode="lines",
            line=dict(color="#2196F3", width=1.5)
        )
    )

    # 3. Linea Macro Swing Grafica
    if swing:
        fig.add_trace(
            go.Scatter(
                x=[swing.start_index, swing.end_index],
                y=[swing.start_price, swing.end_price],
                mode="lines+markers",
                name="Macro Swing",
                line=dict(color="#FF9800", width=3.5),
                marker=dict(size=8, color="#FB8C00")
            )
        )

    # 4. Livelli del Volume Profile (POC, VAH, VAL)
    if profile:
        add_context_lines(fig, profile)

    # 5. Estensione Livelli Fibonacci
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
                annotation_font=dict(color="#90A4AE", size=9)
            )

    # 6. Tracciamento Volume Profile Orizzontale a destra
    if profile and show_volume_profile:
        add_volume_profile(fig, profile)

    # 7. Layout con 2 Assi Ordinati (Prezzo 0-78% della larghezza, Volumi 78-100%)
    fig.update_layout(
        title=dict(
            text="POC Macro Swing & Market Structure Analysis",
            font=dict(size=18, bold=True)
        ),
        height=900,
        template="plotly_white",
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        
        # Asse principale (Candele)
        xaxis=dict(
            domain=[0, 0.78],
            rangeslider_visible=False,
            title="Data"
        ),
        
        # Asse secondario indipendente sovrapposto per il Volume Profile
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


# ==========================================================
# TEST DI SINTASSI
# ==========================================================
if __name__ == "__main__":
    print("Modulo 'charts.py' compilato con successo con assi X indipendenti ed ottimizzati.")