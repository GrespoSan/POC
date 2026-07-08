from __future__ import annotations

import streamlit as st
import pandas as pd

from data import load_ticker
from screening import run_screening
from zigzag import detect_macro_swing
from volume_profile import calculate_volume_profile
from charts import create_chart


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="POC Macro Swing Screener",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 POC Macro Swing Screener")
st.caption("Ricerca titoli con prezzo vicino al Point Of Control del Macro Swing")


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.header("Caricamento mercato")

uploaded_file = st.sidebar.file_uploader(
    "Carica lista ticker (.txt)",
    type=["txt"]
)

tickers = []

if uploaded_file is not None:
    content = uploaded_file.read().decode("utf-8")
    tickers = [
        line.strip().upper()
        for line in content.splitlines()
        if line.strip()
    ]
    st.sidebar.success(f"✅ {len(tickers)} ticker caricati")
else:
    st.sidebar.info("📂 Carica un file ticker per iniziare")


st.sidebar.divider()

st.sidebar.header("Parametri Screener")

lookback = st.sidebar.number_input(
    "Periodo storico",
    min_value=100,
    max_value=2000,
    value=500
)

swing_window = st.sidebar.slider(
    "Pivot Window",
    min_value=2,
    max_value=20,
    value=5
)

atr_period = st.sidebar.slider(
    "ATR Period",
    min_value=5,
    max_value=50,
    value=14
)

min_atr_ratio = st.sidebar.slider(
    "Min ATR Ratio",
    min_value=0.5,
    max_value=5.0,
    value=1.5,
    step=0.1
)

poc_tolerance = st.sidebar.slider(
    "Distanza massima dal POC (%)",
    min_value=0.5,
    max_value=10.0,
    value=3.0,
    step=0.5
)

ema_period = st.sidebar.number_input(
    "EMA Period",
    min_value=20,
    max_value=300,
    value=200
)


# ==========================================================
# RUN SCREENING
# ==========================================================

if st.button("🚀 Avvia Screener"):

    if not tickers:
        st.warning("⚠️ Caricare prima un file ticker dalla barra laterale")
        st.stop()

    progress = st.progress(0)
    status = st.empty()

    def update_progress(value):
        progress.progress(value)
        status.write(f"Analisi completata: {int(value * 100)}%")

    with st.spinner("Analisi titoli in corso..."):

        results = run_screening(
            tickers=tickers,
            loader=load_ticker,
            lookback=lookback,
            swing_window=swing_window,
            atr_period=atr_period,
            min_atr_ratio=min_atr_ratio,
            poc_tolerance=poc_tolerance,
            progress_callback=update_progress
        )

    st.session_state.results = results

    progress.empty()
    status.empty()


# ==========================================================
# RESULTS TABLE
# ==========================================================

if "results" in st.session_state:

    results = st.session_state.results

    if results.empty:

        st.warning("Nessun titolo trovato")

    else:

        st.subheader("Risultati Screening")

        st.dataframe(
            results,
            width="stretch",
            height=400
        )

        st.download_button(
            label="📥 Esporta CSV",
            data=results.to_csv(index=False),
            file_name="poc_results.csv",
            mime="text/csv"
        )

        selected = st.selectbox(
            "Seleziona titolo",
            results["Ticker"].tolist()
        )
        if selected:

            st.divider()
            st.subheader(f"Analisi {selected}")

            # ======================================================
            # CARICAMENTO DATI
            # ======================================================

            df = load_ticker(selected, lookback)

            swing = detect_macro_swing(
                df,
                window=swing_window,
                atr_period=atr_period,
                min_atr_ratio=min_atr_ratio
            )

            if swing is None:

                st.warning("Macro Swing non disponibile")

            else:

                # ======================================================
                # VOLUME PROFILE
                # ======================================================
                # Lo swing NON viene modificato.
                # Il Volume Profile viene invece calcolato dal pivot
                # iniziale fino all'ultima candela disponibile,
                # replicando la logica dell'Anchored VP di TradingView.
                # ======================================================

                profile_df = swing.profile_data(df)

                profile = calculate_volume_profile(profile_df)

                if profile is None:
                    st.error("Impossibile calcolare il Volume Profile.")

                else:

                    fig = create_chart(
                        df=df,
                        swing=swing,
                        profile=profile,
                        ema_period=ema_period
                    )

                    st.plotly_chart(
                        fig,
                        width="stretch"
                    )

                    col1, col2, col3, col4 = st.columns(4)

                    col1.metric(
                        "Trend",
                        swing.direction
                    )

                    col2.metric(
                        "Swing %",
                        f"{swing.move_pct:.2f}%"
                    )

                    col3.metric(
                        "POC",
                        f"{profile.poc:.2f}"
                    )

                    col4.metric(
                        "Score",
                        f"{swing.score:.1f}"
                    )