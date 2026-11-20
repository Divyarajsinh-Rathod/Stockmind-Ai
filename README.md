# STOCKMIND AI – Intelligent Stock Market Analytics

**MSc AI/ML Capstone Project | Course: MAI511-2 (Advanced Machine Learning)**  
**Student Name:** Divyarajsinh Rathod  
**Register Number:** 2648542  

## Overview
STOCKMIND AI is an end-to-end machine learning stock market analytics and price prediction platform built in Python and Streamlit, integrating all 5 units of the `MAI511-2` Advanced Machine Learning syllabus.

## Installation & Running Locally (macOS / VS Code)
```bash
cd /Users/rathoddivyarajsinhyashpalsinh/CapstoneAML
/usr/local/bin/python3 -m pip install -r requirements.txt
/usr/local/bin/python3 -m streamlit run app.py
```

## Project Structure
- `app.py`: 11-section Streamlit financial analytics application.
- `src/data_collection.py`: Yahoo Finance (`yfinance`) historical OHLCV fetcher with local CSV caching in `data/`.
- `src/preprocessing.py`: Chronological forward-fill cleaning, 80/20 time-series split, and leakage-free `StandardScaler`.
- `src/feature_engineering.py`: 19 technical indicators (SMA, EMA, RSI, MACD, Volatility, Volume Ratio, Lags).
- `src/regression.py`: Unit 1 — OLS, Gradient Descent, Simple/Multiple/Multivariate/Polynomial Regression, Ridge, Lasso, Decision Tree, Model Tree, and Logistic Regression.
- `src/dimensionality_reduction.py`: Unit 2 — Low Variance Filter, High Correlation Filter, Forward/Backward Selection, mRMR, Factor Analysis, PCA, LDA, ICA, MDS, and Isomap.
- `src/clustering.py`: Unit 3 — Hopkins Statistic, K-Means, K-Medoids (PAM), Agglomerative & Divisive (DIANA) Hierarchical Clustering, BIRCH, Chameleon, Elbow, Silhouette, Davies-Bouldin, and Calinski-Harabasz.
- `src/reinforcement_learning.py`: Unit 4 — K-Armed Bandit, Model-Based Dyna-Q, and Model-Free TD Q-Learning trading simulation.
- `src/neural_network.py`: Unit 5 — Single Neuron Binary/Bipolar Sigmoid & ReLU calculator (Lab Ex 9), Single-Layer Perceptron, and MLP Backpropagation classifier (Lab Ex 10).
- `src/prediction.py`: End-to-end next-day closing price estimator and multi-day statistical projection.
- `src/visualization.py`: Dark-themed Plotly financial charts and Matplotlib hierarchical dendrograms.
