```python
"""
charts.py
=========

Grafici avanzati per POC Screener.

Include:

- Candlestick
- Macro Swing
- Volume Profile
- POC
- Value Area
- EMA200
- Fibonacci


Workflow


DataFrame OHLCV

        |
        ▼

Plotly Figure

        |
        ▼

Streamlit Chart
"""


from __future__ import annotations


import numpy as np
import pandas as pd

import plotly.graph_objects as go



# ==========================================================
# EMA
# ==========================================================


def calculate_ema(
    df: pd.DataFrame,
    period: int = 200
):

    """
    Calcola EMA.
    """

    return (
        df["Close"]
        .ewm(
            span=period,
            adjust=False
        )
        .mean()
    )



# ==========================================================
# FIBONACCI
# ==========================================================


def fibonacci_levels(
    swing
):

    """
    Calcola livelli Fibonacci
    del macro swing.
    """

    high = max(
        swing.start_price,
        swing.end_price
    )


    low = min(
        swing.start_price,
        swing.end_price
    )


    diff = high - low


    levels = {

        "0.0": high,

        "23.6": high - diff*0.236,

        "38.2": high - diff*0.382,

        "50.0": high - diff*0.5,

        "61.8": high - diff*0.618,

        "78.6": high - diff*0.786,

        "100": low

    }


    return levels



# ==========================================================
# HORIZONTAL LINE
# ==========================================================


def add_horizontal_line(
    fig,
    y,
    name,
):

    """
    Aggiunge linea prezzo.
    """

    fig.add_hline(

        y=y,

        line_dash="dash",

        annotation_text=name

    )



# ==========================================================
# VOLUME PROFILE TRACE
# ==========================================================


def add_volume_profile(
    fig,
    profile,
    x_position
):

    """
    Disegna volume profile orizzontale.
    """

    max_volume = (
        profile.volumes.max()
    )


    widths = (
        profile.volumes /
        max_volume *
        30
    )


    fig.add_trace(

        go.Bar(

            x=widths,

            y=profile.prices,

            orientation="h",

            name="Volume Profile",

            opacity=0.35,

            hovertemplate=

            "Price %{y}<br>" +

            "Volume %{x}"

        )

    )



# ==========================================================
# MAIN CHART FUNCTION
# ==========================================================


def create_chart(
    df: pd.DataFrame,
    swing=None,
    profile=None,
    show_volume_profile=True,
    show_fibonacci=True,
    ema_period=200
):

    """
    Crea grafico completo.
    """



    fig = go.Figure()



    # ------------------------------------------------------
    # Candlestick
    # ------------------------------------------------------

    fig.add_trace(

        go.Candlestick(

            x=df.index,

            open=df["Open"],

            high=df["High"],

            low=df["Low"],

            close=df["Close"],

            name="Price"

        )

    )



    # ------------------------------------------------------
    # EMA
    # ------------------------------------------------------

    ema = calculate_ema(
        df,
        ema_period
    )


    fig.add_trace(

        go.Scatter(

            x=df.index,

            y=ema,

            name=f"EMA {ema_period}",

            mode="lines"

        )

    )



    # ------------------------------------------------------
    # Macro Swing
    # ------------------------------------------------------

    if swing:


        fig.add_trace(

            go.Scatter(

                x=[

                    swing.start_index,

                    swing.end_index

                ],

                y=[

                    swing.start_price,

                    swing.end_price

                ],

                mode="lines+markers",

                name="Macro Swing",

                line=dict(
                    width=3
                )

            )

        )



    # ------------------------------------------------------
    # POC / Value Area
    # ------------------------------------------------------

    if profile:


        add_horizontal_line(
            fig,
            profile.poc,
            "POC"
        )


        add_horizontal_line(
            fig,
            profile.val,
            "VAL"
        )


        add_horizontal_line(
            fig,
            profile.vah,
            "VAH"
        )



    # ------------------------------------------------------
    # Fibonacci
    # ------------------------------------------------------

    if swing and show_fibonacci:


        fibs = fibonacci_levels(
            swing
        )


        for name, price in fibs.items():

            fig.add_hline(

                y=price,

                line_dash="dot",

                annotation_text=
                f"Fib {name}"

            )



    # ------------------------------------------------------
    # Volume Profile
    # ------------------------------------------------------

    if profile and show_volume_profile:


        add_volume_profile(

            fig,

            profile,

            len(df)

        )



    # ------------------------------------------------------
    # Layout
    # ------------------------------------------------------

    fig.update_layout(

        title="POC Macro Swing Analysis",

        height=800,

        xaxis_rangeslider_visible=False,

        hovermode="x unified",

        template="plotly_white"

    )



    return fig



# ==========================================================
# TEST
# ==========================================================


if __name__ == "__main__":


    from data import load_ticker

    from zigzag import detect_macro_swing

    from volume_profile import calculate_volume_profile



    df = load_ticker(
        "AAPL",
        500
    )


    swing = detect_macro_swing(
        df
    )


    if swing:

        swing_df = swing.data(
            df
        )


        profile = calculate_volume_profile(
            swing_df
        )


        fig = create_chart(

            df,

            swing,

            profile

        )


        fig.show()
```
