
🔍 Overview

This project is a real-time analytics dashboard built using Streamlit that analyzes intraday oil market data. 
It combines OHLC price data with trade-and-sale (TAS) data to extract volume delta, momentum, and market structure signals.

The system generates actionable signals (Long, Short, Trap, Reversal) using a rule-based signal engine driven by volume, delta, and divergence analysis.

⚙️ Features
  Real-time data ingestion and refresh (60 seconds)
  Multi-chart visualization using Plotly
  Volume delta & delta-diff analytics
  Custom signal engine with:
  Trap detection
  Momentum-based entries
  Early exit signals
  Market structure classification (Bullish, Bearish, Neutral, Absorbing)
  Interactive time window slider synced across charts
  Signal history tracking
🧠 Signal Logic (Core Analytics)

The signal engine uses:

    Volume thresholds (relative to rolling average)
    Delta (buy vs sell imbalance)
    Rate of Change (ROC) of delta
    Price vs delta divergence

Signals include:

    ENTER_LONG / ENTER_SHORT
    EXIT (Early / Confirmed)
    TRAP (absorption detection)
📊 Tech Stack
  Python
  Streamlit (UI)
  Plotly (charts)
  Pandas (data processing)
  REST APIs for OHLC & TAS data
🚀 How to Run
pip install streamlit streamlit-autorefresh plotly pandas requests
streamlit run monitor.py
📈 Data Pipeline
Fetch OHLC data (1-min resolution)
Fetch TAS (trade-level data)
Aggregate TAS → volume delta
Merge datasets using time alignment
Compute derived metrics:
Delta
ROC of delta
Delta difference
Run signal engine
Visualize in synchronized charts
📌 Key Insights
Detects institutional activity using volume + delta imbalance
Identifies absorption zones (traps)
Avoids lagging indicators (purely data-driven, no RSI/MACD)
Designed for intraday decision-making


🔮 Future Improvements
Backtesting engine
ML-based signal optimization
Multi-asset support
Alert integrations
