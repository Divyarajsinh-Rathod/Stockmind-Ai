"""
Interactive Learning Content, Lucide SVG Icons & Glossary Module (src/learning_content.py)
Provides beginner-friendly explanations, faculty presentation scripts, code explainers,
SVG learning diagrams, and a searchable ML & Financial Glossary for STOCKMIND AI.
"""

# Crisp Lucide-style inline SVG icons (no random emojis)
_SVG_PATHS = {
    "dashboard": '<rect x="3" y="3" width="7" height="9" rx="1"/><rect x="14" y="3" width="7" height="5" rx="1"/><rect x="14" y="12" width="7" height="9" rx="1"/><rect x="3" y="16" width="7" height="5" rx="1"/>',
    "stock": '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/>',
    "sliders": '<line x1="4" x2="4" y1="21" y2="14"/><line x1="4" x2="4" y1="10" y2="3"/><line x1="12" x2="12" y1="21" y2="12"/><line x1="12" x2="12" y1="8" y2="3"/><line x1="20" x2="20" y1="21" y2="16"/><line x1="20" x2="20" y1="12" y2="3"/><line x1="2" x2="6" y1="14" y2="14"/><line x1="10" x2="14" y1="8" y2="8"/><line x1="18" x2="22" y1="16" y2="16"/>',
    "regression": '<path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/>',
    "layers": '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
    "cluster": '<circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" x2="15.42" y1="13.51" y2="17.49"/><line x1="15.41" x2="8.59" y1="6.51" y2="10.49"/>',
    "bot": '<path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/>',
    "brain": '<path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-2.54"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-2.54"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "award": '<circle cx="12" cy="8" r="6"/><path d="M15.477 12.89 17 22l-5-3-5 3 1.523-9.11"/>',
    "book": '<path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"/>',
    "compass": '<circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/>',
    "code": '<polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "workflow": '<rect width="8" height="8" x="3" y="3" rx="2"/><path d="M7 11v4a2 2 0 0 0 2 2h4"/><rect width="8" height="8" x="13" y="13" rx="2"/>',
}


def get_icon(name: str, color: str = "#38BDF8", size: int = 20) -> str:
    """Returns a crisp inline Lucide SVG icon."""
    path = _SVG_PATHS.get(name, _SVG_PATHS["info"])
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:8px;">'
        f'{path}</svg>'
    )


# 10-Point Structured Algorithm Explanations for Learning & Faculty Modes
ALGORITHM_EXPLANATIONS = {
    "Simple Linear Regression": {
        "what": "A foundational supervised algorithm that fits a single straight line (y = mx + b) between one input feature and one numerical target.",
        "why_need": "To establish a transparent, interpretable baseline before adding complex features.",
        "why_project": "Tests how much of tomorrow's closing price can be explained using only today's closing price.",
        "how_works": "1. Takes today's Close (x) and tomorrow's Close (y). 2. Finds slope (m) and intercept (b) that minimize squared prediction errors. 3. Projects tomorrow's price along that line.",
        "data_used": "Single feature: `Close_t` predicting `Target_Next_Close`.",
        "internal": "Computes Ordinary Least Squares (OLS) closed-form solution: beta = (X^T X)^(-1) X^T y.",
        "output": "Continuous predicted closing price for day t+1.",
        "interpret": "High R² near 0.90+ occurs because stock prices are autocorrelated; compare MAE/RMSE against the Naive Baseline.",
        "advantages": "Zero hyperparameters, instant training, easy to explain mathematically.",
        "limitations": "Ignores momentum, volume, RSI, and nonlinear market shifts.",
        "faculty_note": "We use Simple Linear Regression as our Unit 1 baseline to demonstrate OLS estimation before introducing multi-feature and regularized models.",
    },
    "Multiple Linear Regression": {
        "what": "An extension of Linear Regression that combines multiple input features simultaneously using weighted linear summation.",
        "why_need": "Stock prices depend on trend, momentum, volatility, and volume together—not just yesterday's price.",
        "why_project": "Combines all 19 engineered technical indicators (SMA, EMA, RSI, MACD, Volatility, Volume Ratio) to predict tomorrow's Close.",
        "how_works": "1. Standardizes all 19 features on training data. 2. Assigns one weight w_i per indicator. 3. Sums weighted features plus bias to output Target_Next_Close.",
        "data_used": "All 19 scaled technical indicators at day t.",
        "internal": "Solves the normal equations via QR/SVD matrix decomposition in Scikit-learn's `LinearRegression`.",
        "output": "Continuous price forecast plus interpretable feature coefficients.",
        "interpret": "Large positive/negative weights on correlated moving averages signal multicollinearity—motivating Ridge and Lasso.",
        "advantages": "Captures multi-indicator linear relationships in a single equation.",
        "limitations": "Sensitive to multicollinearity between correlated features like SMA_10, SMA_20, and EMA_12.",
        "faculty_note": "Multiple Linear Regression combines all 19 indicators, and its large opposing weights on correlated moving averages directly motivate L2/L1 regularization.",
    },
    "Polynomial Regression": {
        "what": "Transforms input features into squared or cubic interaction terms (x, x², x_i * x_j) and fits a linear model on those expanded features.",
        "why_need": "Real market behavior is often curved or nonlinear, especially during high-volatility breakouts.",
        "why_project": "Captures nonlinear interactions among Close, SMA_10, RSI_14, MACD, and Volatility_20.",
        "how_works": "1. Expands 5 core indicators to degree 2 or 3. 2. Applies Ridge stabilization. 3. Learns curved decision surfaces.",
        "data_used": "5 core indicators (`Close`, `SMA_10`, `RSI_14`, `MACD`, `Volatility_20`).",
        "internal": "`PolynomialFeatures(degree=d)` generates interaction columns, followed by regularized linear fitting.",
        "output": "Continuous nonlinear price estimate.",
        "interpret": "Watch the Train vs Test R² gap: degree 3 often overfits noise compared to degree 2.",
        "advantages": "Models curved relationships while remaining a linear-in-parameters solver.",
        "limitations": "Feature count explodes combinatorially with degree, increasing overfitting risk.",
        "faculty_note": "We demonstrate degree-2 and degree-3 Polynomial Regression (Unit 1 Lab Ex 2) to show how nonlinear expansion helps capture curvature but increases variance if degree is too high.",
    },
    "Ridge Regression": {
        "what": "Linear Regression with an L2 regularization penalty (alpha * sum(w_j^2)) added to the loss function.",
        "why_need": "Technical indicators like SMA_10, SMA_20, and EMA_12 move together (multicollinearity), causing unregularized weights to explode.",
        "why_project": "Serves as our primary stable continuous stock price predictor across the Dashboard and Prediction Engine.",
        "how_works": "1. Penalizes large squared weights during training. 2. Shrinks correlated indicator weights smoothly toward zero without deleting them.",
        "data_used": "All 19 standardized technical indicators (scaled strictly on X_train).",
        "internal": "Minimizes ||y - Xw||^2 + alpha * ||w||^2, ensuring the matrix (X^T X + alpha*I) is always invertible.",
        "output": "Stable continuous price prediction with lower test variance.",
        "interpret": "Compare Ridge coefficients against OLS coefficients in Section 4 to see weight shrinkage in action.",
        "advantages": "Strongly reduces overfitting and handles correlated financial features gracefully.",
        "limitations": "Keeps all features active (weights approach zero but rarely equal exact zero).",
        "faculty_note": "Ridge Regression is our most reliable linear predictor because L2 regularization stabilizes weights across highly correlated technical indicators.",
    },
    "Lasso Regression": {
        "what": "Linear Regression with an L1 regularization penalty (alpha * sum(|w_j|)) that can shrink redundant weights to exact zero.",
        "why_need": "To automatically discard noisy or redundant indicators while training.",
        "why_project": "Demonstrates embedded feature selection on our 19 stock market indicators.",
        "how_works": "1. Adds absolute-value penalty to MSE loss. 2. Uses coordinate descent to force weak feature coefficients to 0.0000.",
        "data_used": "All 19 standardized technical indicators.",
        "internal": "Solves min (1/2n)||y - Xw||^2 + alpha * ||w||_1 via iterative coordinate descent.",
        "output": "Sparse price prediction model with a subset of non-zero features.",
        "interpret": "Features with `0.0000` in the coefficient table were automatically eliminated by L1 regularization.",
        "advantages": "Produces simpler, sparse, more interpretable models.",
        "limitations": "When two features are nearly identical, Lasso arbitrarily picks one and zeros the other.",
        "faculty_note": "Lasso demonstrates L1 regularization from Unit 1, acting as automatic feature selection by setting redundant indicator weights to exact zero.",
    },
    "Decision Tree & Model Tree Regression": {
        "what": "Decision Trees split data into rectangular regions predicting a constant average per leaf; Model Trees fit a local linear regression inside each leaf.",
        "why_need": "To demonstrate non-parametric tree splitting and the Bias-Variance / Overfitting trade-off.",
        "why_project": "Contrasts an unconstrained overfit tree (Train R²~1.0, poor Test R²), a pruned tree (max_depth=4), and a Model Tree.",
        "how_works": "1. Splits features at thresholds minimizing MSE. 2. Standard trees return leaf mean; Model Trees fit a Ridge model in each leaf.",
        "data_used": "All 19 scaled technical features.",
        "internal": "CART recursive binary splitting (`DecisionTreeRegressor`) + leaf-wise `Ridge` fitting in `SimpleLinearModelTree`.",
        "output": "Piecewise constant (Decision Tree) or piecewise linear (Model Tree) price forecasts.",
        "interpret": "Standard trees cannot extrapolate beyond historical training price ranges; Model Trees overcome this limitation.",
        "advantages": "Visualizes overfitting clearly and handles nonlinear market regimes.",
        "limitations": "Standard regression trees fail when test prices rise above historical training maximums.",
        "faculty_note": "Comparing an unconstrained tree vs pruned tree proves overfitting visually, and our Model Tree fixes tree extrapolation by fitting linear equations in each leaf.",
    },
    "Logistic Regression": {
        "what": "A supervised classification algorithm that passes a linear score through a Sigmoid function to predict binary class probabilities (0 or 1).",
        "why_need": "Predicting continuous price (₹2,540) is Regression; predicting market direction (Up vs Down) is Classification.",
        "why_project": "Predicts whether tomorrow's closing price will be higher (1) or lower (0) than today's closing price.",
        "how_works": "1. Computes z = w^T x + b. 2. Applies P(Up) = 1 / (1 + e^-z). 3. Outputs Up (1) if P >= 0.5 else Down (0).",
        "data_used": "All 19 scaled technical features with `Target_Direction` (0/1).",
        "internal": "Minimizes binary cross-entropy (log-loss) using L-BFGS optimization.",
        "output": "Binary directional class (0 or 1) and probability score.",
        "interpret": "Evaluated using Accuracy, Precision, Recall, and F1-Score—never RMSE or R².",
        "advantages": "Outputs calibrated probabilities and clearly separates classification from regression.",
        "limitations": "Assumes a linear boundary between bullish and bearish days.",
        "faculty_note": "Crucially, we never use Logistic Regression to predict continuous stock prices—it is strictly used for binary Up/Down direction classification.",
    },
    "Feature Selection (Filters, Wrappers & mRMR)": {
        "what": "Techniques that select the most informative subset of original features without transforming their meaning.",
        "why_need": "Redundant or low-variance features add noise, slow training, and hurt generalization.",
        "why_project": "Compares Low Variance Filter, High Correlation Filter, Forward Selection, Backward Elimination, and mRMR on X_train.",
        "how_works": "Filters rank statistical properties (variance, correlation, F-score); Wrappers test subsets via cross-validated Ridge regression.",
        "data_used": "19 technical features strictly from the training split (`X_train`).",
        "internal": "`VarianceThreshold`, Pearson correlation matrix, `SequentialFeatureSelector`, and greedy mRMR F-score/redundancy ratio.",
        "output": "Lists of kept vs removed features and before-vs-after RMSE comparison.",
        "interpret": "Removing collinear features often keeps Test R² intact while cutting compute time and model complexity.",
        "advantages": "Preserves original financial feature names for easy interpretation.",
        "limitations": "Wrapper methods (SFS/SBS) are computationally slower than filters.",
        "faculty_note": "All feature selection is performed strictly on `X_train` so test-set information never leaks into feature choices.",
    },
    "PCA vs LDA vs ICA & Manifolds": {
        "what": "Projection methods that compress 19 features into 2–5 lower-dimensional components.",
        "why_need": "To eliminate multicollinearity and visualize high-dimensional market regimes in 2D.",
        "why_project": "Contrasts Unsupervised PCA/ICA/FA/MDS/Isomap against Supervised LDA (which uses Bullish/Neutral/Bearish labels).",
        "how_works": "PCA maximizes orthogonal variance; LDA maximizes class separation; ICA finds statistically independent sources; MDS/Isomap preserve distances.",
        "data_used": "Standardized `X_train` (plus `Target_Trend_Class` for supervised LDA only).",
        "internal": "Eigenvalue/SVD decomposition (PCA/LDA), FastICA, FactorAnalysis, and geodesic/stress manifold optimization.",
        "output": "PCA Scree plot (explained variance) and 2D regime scatter plots.",
        "interpret": "Check how many PCs are needed to capture 90% of total market variance.",
        "advantages": "Compresses 19 correlated indicators into uncorrelated components.",
        "limitations": "Transformed components (PC1, PC2) are linear combinations rather than single named indicators.",
        "faculty_note": "Unit 2 highlights the core distinction between unsupervised PCA (no labels) and supervised LDA (uses class labels to separate market regimes).",
    },
    "Clustering (K-Means, K-Medoids, Hierarchical, BIRCH, Chameleon)": {
        "what": "Unsupervised learning methods that group stocks with similar historical risk-return profiles without labels.",
        "why_need": "Investors need to identify behavioral groups (High-Growth Momentum vs Defensive Blue-Chip) for diversification.",
        "why_project": "Clusters 12 Indian (NSE) and US (NASDAQ) stocks using Return, Volatility, Sharpe Ratio, RSI, Intraday Range, and Max Drawdown.",
        "how_works": "K-Means uses centroids; K-Medoids (PAM) picks actual stocks as centers; Agglomerative/DIANA build trees; BIRCH uses CF-trees.",
        "data_used": "Cross-sectional 1-year behavioral metrics across 12 equities.",
        "internal": "Hopkins tendency check -> Elbow WCSS & Silhouette/DBI/CHI scoring -> 6 clustering algorithms.",
        "output": "Cluster assignments, dendrogram, medoid tickers, and risk-return scatter map.",
        "interpret": "Clustering discovers market structure—it does NOT predict tomorrow's stock price.",
        "advantages": "Reveals portfolio diversification opportunities across Indian and US markets.",
        "limitations": "Sensitive to feature scaling and chosen number of clusters k.",
        "faculty_note": "We first verify clusterability via the Hopkins Statistic, determine optimal k via Elbow and Silhouette scores, and compare 6 Unit 3 algorithms.",
    },
    "Reinforcement Learning (Bandit, TD Q-Learning & Model-Based Dyna-Q)": {
        "what": "An agent learns by interacting with an environment, choosing Buy/Sell/Hold actions to maximize cumulative reward.",
        "why_need": "Sequential trading decisions depend on current market state and delayed future outcomes.",
        "why_project": "Simulates an educational trading agent across 18 discrete market states (Trend x RSI x Volatility) using virtual capital.",
        "how_works": "Bandit tests 4 single-state rules; Q-Learning updates Q(s,a) from experience; Model-Based Dyna-Q also learns transition/reward models for planning.",
        "data_used": "Discretized historical market states and next-day percentage returns.",
        "internal": "Bellman TD update: Q(s,a) <- Q(s,a) + alpha * [r + gamma*max Q(s',a') - Q(s,a)] + planning steps.",
        "output": "Simulated portfolio curve vs Buy-and-Hold, learned Q-table policy, and action counts.",
        "interpret": "Demonstrates exploration vs exploitation and model-based vs model-free learning—not a live trading bot.",
        "advantages": "Directly models sequential decision-making and transaction friction.",
        "limitations": "Discretized states simplify complex markets; simulated returns do not guarantee future profits.",
        "faculty_note": "This section implements Unit 4 Lab Exercise 8, contrasting Model-Free TD Q-Learning against Model-Based Dyna-Q planning.",
    },
    "Neural Networks (Single Neuron, Perceptron & MLP Backpropagation)": {
        "what": "Layered networks of artificial neurons inspired by biological synapses that learn nonlinear patterns via backpropagation.",
        "why_need": "Single-layer linear models cannot solve nonlinearly separable market patterns.",
        "why_project": "Lab Ex 9 demonstrates Binary/Bipolar Sigmoid & ReLU on one neuron; Lab Ex 10 trains an MLP to classify Up vs Down days.",
        "how_works": "Forward pass computes weighted sums + activation; Backpropagation computes loss gradients layer-by-layer to update weights.",
        "data_used": "19 standardized technical indicators predicting `Target_Direction` (0/1).",
        "internal": "Epoch-by-epoch `partial_fit` with Adam optimizer recording Training & Validation cross-entropy loss.",
        "output": "Neuron activation values, training/validation loss curves, confusion matrix, and classification report.",
        "interpret": "Diverging training vs validation loss indicates overfitting; adjust hidden layers or epochs.",
        "advantages": "Learns complex nonlinear combinations of indicators automatically.",
        "limitations": "Requires careful scaling and tuning to avoid memorizing noisy daily market fluctuations.",
        "faculty_note": "We connect Lab Ex 9 (single neuron Binary vs Bipolar Sigmoid math) directly to Lab Ex 10 (full MLP with Backpropagation).",
    },
}


# Code Snippets & Key Function Explainers
CODE_EXPLAINERS = {
    "preprocessing": {
        "title": "Chronological Split & Leakage-Free Scaling (src/preprocessing.py)",
        "snippet": """split_idx = int(len(valid_df) * train_ratio)
train_df = valid_df.iloc[:split_idx]
test_df = valid_df.iloc[split_idx:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)""",
        "functions": [
            ("fit_transform()", "Learns mean & standard deviation ONLY from training data, then scales training data."),
            ("transform()", "Applies the already-learned training mean & std to unseen test data (prevents data leakage)."),
            ("Chronological iloc[:split_idx]", "Splits time-series strictly by date order without random shuffling."),
        ],
    },
    "regression": {
        "title": "Training & Evaluating Regression Models (src/regression.py)",
        "snippet": """ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
y_test_pred = ridge.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))""",
        "functions": [
            ("LinearRegression() / Ridge()", "Creates linear OLS or L2-regularized regression model instances."),
            ("fit(X, y)", "Trains the model by finding optimal weights that minimize prediction error on training data."),
            ("predict(X)", "Uses learned weights to estimate next-day closing prices on unseen test data."),
        ],
    },
    "dim_reduction": {
        "title": "PCA & Supervised LDA Projection (src/dimensionality_reduction.py)",
        "snippet": """pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_train_scaled, y_trend_train)""",
        "functions": [
            ("PCA(n_components=2)", "Unsupervised: compresses 19 features into 2 orthogonal axes capturing maximum variance."),
            ("LinearDiscriminantAnalysis()", "Supervised: uses Bearish/Neutral/Bullish labels to maximize class separation."),
        ],
    },
    "clustering": {
        "title": "K-Means & Silhouette Evaluation (src/clustering.py)",
        "snippet": """kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
score = silhouette_score(X_scaled, labels)""",
        "functions": [
            ("KMeans(n_clusters=k)", "Partitions stocks into k behavioral clusters around centroid averages."),
            ("fit_predict(X_scaled)", "Finds cluster centers and assigns each stock its cluster ID in one step."),
            ("silhouette_score()", "Measures how well-separated and compact the clusters are (-1 to +1)."),
        ],
    },
    "neural_network": {
        "title": "MLP Classifier with Backpropagation (src/neural_network.py)",
        "snippet": """mlp = MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu', warm_start=True)
for epoch in range(1, max_epochs + 1):
    mlp.partial_fit(X_sub_tr, y_sub_tr, classes=[0, 1])""",
        "functions": [
            ("MLPClassifier()", "Builds a Multilayer Perceptron neural network with hidden layers and backpropagation."),
            ("partial_fit()", "Trains one epoch at a time so we can track Training & Validation log-loss curves."),
        ],
    },
}


# Inline SVG Visual Learning Diagrams (Clean, Educational, Contrast-Safe)
def get_pipeline_svg_diagram() -> str:
    return """
    <div style="background:rgba(15,23,42,0.65);border:1px solid rgba(56,189,248,0.25);border-radius:12px;padding:16px;margin:12px 0;overflow-x:auto;">
      <svg viewBox="0 0 920 110" width="100%" height="110" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="#38BDF8"/>
          </marker>
        </defs>
        <rect x="10" y="22" width="150" height="62" rx="10" fill="#1E293B" stroke="#38BDF8" stroke-width="1.5"/>
        <text x="85" y="48" fill="#F8FAFC" font-size="12" font-weight="bold" text-anchor="middle">1. Yahoo Finance</text>
        <text x="85" y="66" fill="#94A3B8" font-size="10" text-anchor="middle">OHLCV Data + Cache</text>

        <line x1="160" y1="53" x2="192" y2="53" stroke="#38BDF8" stroke-width="2" marker-end="url(#arrow)"/>

        <rect x="195" y="22" width="150" height="62" rx="10" fill="#1E293B" stroke="#818CF8" stroke-width="1.5"/>
        <text x="270" y="48" fill="#F8FAFC" font-size="12" font-weight="bold" text-anchor="middle">2. Preprocessing</text>
        <text x="270" y="66" fill="#94A3B8" font-size="10" text-anchor="middle">Forward-Fill &amp; 80/20 Split</text>

        <line x1="345" y1="53" x2="377" y2="53" stroke="#38BDF8" stroke-width="2" marker-end="url(#arrow)"/>

        <rect x="380" y="22" width="155" height="62" rx="10" fill="#1E293B" stroke="#A855F7" stroke-width="1.5"/>
        <text x="457" y="48" fill="#F8FAFC" font-size="12" font-weight="bold" text-anchor="middle">3. Feature Engineering</text>
        <text x="457" y="66" fill="#94A3B8" font-size="10" text-anchor="middle">19 Indicators + Shift(-1)</text>

        <line x1="535" y1="53" x2="567" y2="53" stroke="#38BDF8" stroke-width="2" marker-end="url(#arrow)"/>

        <rect x="570" y="22" width="155" height="62" rx="10" fill="#1E293B" stroke="#10B981" stroke-width="1.5"/>
        <text x="647" y="48" fill="#F8FAFC" font-size="12" font-weight="bold" text-anchor="middle">4. 5-Unit ML Engine</text>
        <text x="647" y="66" fill="#94A3B8" font-size="10" text-anchor="middle">Reg, PCA, Cluster, RL, MLP</text>

        <line x1="725" y1="53" x2="757" y2="53" stroke="#38BDF8" stroke-width="2" marker-end="url(#arrow)"/>

        <rect x="760" y="22" width="150" height="62" rx="10" fill="#1E293B" stroke="#F59E0B" stroke-width="1.5"/>
        <text x="835" y="48" fill="#F8FAFC" font-size="12" font-weight="bold" text-anchor="middle">5. Forecast &amp; Eval</text>
        <text x="835" y="66" fill="#94A3B8" font-size="10" text-anchor="middle">MAE, RMSE, R², Acc</text>
      </svg>
    </div>
    """


def get_ann_svg_diagram(hidden_str: str = "(32, 16)", activation: str = "RELU") -> str:
    return f"""
    <div style="background:rgba(15,23,42,0.65);border:1px solid rgba(168,85,247,0.3);border-radius:12px;padding:14px;margin:10px 0;">
      <svg viewBox="0 0 720 190" width="100%" height="185" xmlns="http://www.w3.org/2000/svg">
        <text x="90" y="20" fill="#38BDF8" font-size="12" font-weight="bold" text-anchor="middle">Input Layer (19 Features)</text>
        <text x="290" y="20" fill="#A855F7" font-size="12" font-weight="bold" text-anchor="middle">Hidden Layers {hidden_str}</text>
        <text x="530" y="20" fill="#10B981" font-size="12" font-weight="bold" text-anchor="middle">Output Layer (1 Node)</text>

        <!-- Connections -->
        <g stroke="rgba(148,163,184,0.35)" stroke-width="1.2">
          <line x1="110" y1="50" x2="250" y2="55"/><line x1="110" y1="50" x2="250" y2="100"/><line x1="110" y1="50" x2="250" y2="145"/>
          <line x1="110" y1="100" x2="250" y2="55"/><line x1="110" y1="100" x2="250" y2="100"/><line x1="110" y1="100" x2="250" y2="145"/>
          <line x1="110" y1="150" x2="250" y2="55"/><line x1="110" y1="150" x2="250" y2="100"/><line x1="110" y1="150" x2="250" y2="145"/>
          <line x1="270" y1="55" x2="360" y2="75"/><line x1="270" y1="100" x2="360" y2="75"/><line x1="270" y1="145" x2="360" y2="75"/>
          <line x1="270" y1="55" x2="360" y2="125"/><line x1="270" y1="100" x2="360" y2="125"/><line x1="270" y1="145" x2="360" y2="125"/>
          <line x1="380" y1="75" x2="515" y2="100"/><line x1="380" y1="125" x2="515" y2="100"/>
        </g>

        <!-- Input Nodes -->
        <circle cx="95" cy="50" r="14" fill="#1E293B" stroke="#38BDF8" stroke-width="2"/><text x="95" y="54" fill="#F8FAFC" font-size="9" text-anchor="middle">Close</text>
        <circle cx="95" cy="100" r="14" fill="#1E293B" stroke="#38BDF8" stroke-width="2"/><text x="95" y="104" fill="#F8FAFC" font-size="9" text-anchor="middle">RSI</text>
        <circle cx="95" cy="150" r="14" fill="#1E293B" stroke="#38BDF8" stroke-width="2"/><text x="95" y="154" fill="#F8FAFC" font-size="9" text-anchor="middle">MACD</text>

        <!-- Hidden Nodes -->
        <circle cx="260" cy="55" r="14" fill="#1E293B" stroke="#A855F7" stroke-width="2"/>
        <circle cx="260" cy="100" r="14" fill="#1E293B" stroke="#A855F7" stroke-width="2"/>
        <circle cx="260" cy="145" r="14" fill="#1E293B" stroke="#A855F7" stroke-width="2"/>
        <circle cx="370" cy="75" r="14" fill="#1E293B" stroke="#818CF8" stroke-width="2"/>
        <circle cx="370" cy="125" r="14" fill="#1E293B" stroke="#818CF8" stroke-width="2"/>
        <text x="315" y="175" fill="#94A3B8" font-size="10" text-anchor="middle">Activation: {activation} | Backpropagation (Adam)</text>

        <!-- Output Node -->
        <circle cx="530" cy="100" r="18" fill="#1E293B" stroke="#10B981" stroke-width="2.5"/>
        <text x="530" y="104" fill="#10B981" font-size="10" font-weight="bold" text-anchor="middle">Up / Dn</text>
        <text x="635" y="104" fill="#E2E8F0" font-size="11">P(Close_t+1 &gt; Close_t)</text>
      </svg>
    </div>
    """


def get_rl_svg_diagram() -> str:
    return """
    <div style="background:rgba(15,23,42,0.65);border:1px solid rgba(16,185,129,0.3);border-radius:12px;padding:14px;margin:10px 0;">
      <svg viewBox="0 0 720 140" width="100%" height="140" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <marker id="rl_arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="#10B981"/>
          </marker>
        </defs>
        <rect x="60" y="38" width="180" height="64" rx="12" fill="#1E293B" stroke="#38BDF8" stroke-width="2"/>
        <text x="150" y="65" fill="#F8FAFC" font-size="13" font-weight="bold" text-anchor="middle">RL Trading Agent</text>
        <text x="150" y="84" fill="#94A3B8" font-size="10" text-anchor="middle">Policy π(s) &amp; Q(s, a) Table</text>

        <rect x="470" y="38" width="190" height="64" rx="12" fill="#1E293B" stroke="#10B981" stroke-width="2"/>
        <text x="565" y="65" fill="#F8FAFC" font-size="13" font-weight="bold" text-anchor="middle">Market Environment</text>
        <text x="565" y="84" fill="#94A3B8" font-size="10" text-anchor="middle">18 States (Trend × RSI × Vol)</text>

        <path d="M 240 52 Q 355 18 466 52" fill="none" stroke="#10B981" stroke-width="2" marker-end="url(#rl_arr)"/>
        <text x="355" y="28" fill="#10B981" font-size="11" font-weight="bold" text-anchor="middle">Action A_t: Buy / Sell / Hold</text>

        <path d="M 470 88 Q 355 125 244 88" fill="none" stroke="#F59E0B" stroke-width="2" marker-end="url(#rl_arr)"/>
        <text x="355" y="124" fill="#F59E0B" font-size="11" font-weight="bold" text-anchor="middle">Next State S_(t+1) + Reward R_t (% Return)</text>
      </svg>
    </div>
    """


# Searchable Financial & Machine Learning Glossary
GLOSSARY_ITEMS = [
    {"Term": "Open / High / Low / Close (OHLC)", "Category": "Stock Market", "Simple": "The four key daily prices of a stock: opening price, highest price, lowest price, and final closing price.", "Technical": "Daily candlestick summary prices recorded at market open, intraday extrema, and market close.", "Project_Use": "Base raw inputs from Yahoo Finance; `Close` is used to build all indicators and the next-day target."},
    {"Term": "Volume", "Category": "Stock Market", "Simple": "Total number of shares traded on a given day.", "Technical": "Aggregate share turnover during a trading session.", "Project_Use": "Used in `Volume_Ratio` to detect unusual institutional buying or selling."},
    {"Term": "Stock Ticker", "Category": "Stock Market", "Simple": "A short unique code identifying a company on a stock exchange (e.g., RELIANCE.NS or AAPL).", "Technical": "Exchange-qualified equity symbol used by Yahoo Finance API.", "Project_Use": "Selected in the sidebar to load Indian (NSE) or US (NASDAQ) stocks."},
    {"Term": "Bullish vs Bearish Market", "Category": "Stock Market", "Simple": "Bullish means prices are rising; Bearish means prices are falling.", "Technical": "Positive vs negative directional market regime.", "Project_Use": "Used as class labels in Logistic Regression, LDA, and MLP directional classification."},
    {"Term": "SMA & EMA (Moving Averages)", "Category": "Technical Indicators", "Simple": "Average stock price over the last N days to smooth out daily ups and downs.", "Technical": "Simple Moving Average (equal weights) vs Exponential Moving Average (higher weight on recent days).", "Project_Use": "Features `SMA_10`, `SMA_20`, `EMA_12`, `EMA_26` capture short- and medium-term trends."},
    {"Term": "RSI (Relative Strength Index)", "Category": "Technical Indicators", "Simple": "A 0-to-100 speedometer showing if a stock rose too fast (overbought >70) or fell too fast (oversold <30).", "Technical": "14-day momentum oscillator comparing average gains to average losses.", "Project_Use": "Feature `RSI_14` used across Regression, RL states, and Clustering."},
    {"Term": "MACD", "Category": "Technical Indicators", "Simple": "Shows whether short-term price momentum is speeding up or slowing down.", "Technical": "Moving Average Convergence Divergence: EMA(12) minus EMA(26) compared with a 9-day Signal line.", "Project_Use": "Features `MACD` and `MACD_Signal` feed ML models and K-Armed Bandit rules."},
    {"Term": "Daily Returns & Rolling Volatility", "Category": "Technical Indicators", "Simple": "Daily Return is today's % gain/loss; Volatility measures how wildly the price swings.", "Technical": "Percentage change in Close and 20-day rolling standard deviation of daily returns.", "Project_Use": "Key risk features (`Daily_Return`, `Volatility_20`) for Clustering and RL state discretization."},
    {"Term": "Features (X) vs Target Variable (y)", "Category": "ML Core", "Simple": "Features are the clues given to the model; Target is the answer we want it to predict.", "Technical": "Independent input matrix X_t (19 indicators) vs dependent label y_(t+1).", "Project_Use": "`Target_Next_Close` (regression) and `Target_Direction` (classification)."},
    {"Term": "Chronological Train-Test Split", "Category": "ML Core", "Simple": "Training on older dates and testing on newer dates without shuffling.", "Technical": "Temporal holdout split preserving time order to prevent look-ahead bias.", "Project_Use": "80% earliest trading days train models; 20% most recent days evaluate generalization."},
    {"Term": "Data Leakage", "Category": "ML Core", "Simple": "Accidentally giving the model future information it would not have in real life.", "Technical": "Using future observations or test-set statistics during training or scaling.", "Project_Use": "Strictly prevented by shifting targets `shift(-1)` and fitting scalers/PCA only on `X_train`."},
    {"Term": "MAE, MSE, RMSE & R² Score", "Category": "Unit 1 (Evaluation)", "Simple": "Scores that tell us how far off our price predictions are (lower error = better; R² closer to 1 = better).", "Technical": "Mean Absolute Error, Mean Squared Error, Root Mean Squared Error, and Coefficient of Determination.", "Project_Use": "Used to benchmark all 10 Regression models in Section 4 & Section 10."},
    {"Term": "Overfitting vs Underfitting", "Category": "Unit 1 (Evaluation)", "Simple": "Overfitting = memorizing past data (fails on new data); Underfitting = too simple to learn the pattern.", "Technical": "High variance (large Train-Test R² gap) vs High bias (poor Train & Test R²).", "Project_Use": "Demonstrated by comparing Unconstrained Decision Tree vs Pruned Tree and Ridge Regression."},
    {"Term": "OLS & Gradient Descent", "Category": "Unit 1 (Regression)", "Simple": "Two ways to find the best-fit line: OLS solves it in one math formula; Gradient Descent takes small steps downhill.", "Technical": "Closed-form normal equation vs iterative first-order optimization of MSE.", "Project_Use": "Compared side-by-side with an epoch loss curve in Section 4."},
    {"Term": "Ridge (L2) vs Lasso (L1)", "Category": "Unit 1 (Regression)", "Simple": "Penalties that keep model weights small; Lasso can even turn useless features completely off (0).", "Technical": "L2 squared-weight shrinkage vs L1 absolute-weight sparsity regularization.", "Project_Use": "Ridge is our primary stable predictor; Lasso demonstrates automatic feature zeroing."},
    {"Term": "PCA vs LDA", "Category": "Unit 2 (Dim. Reduction)", "Simple": "PCA compresses features without knowing labels; LDA compresses features to best separate Up vs Down days.", "Technical": "Unsupervised orthogonal variance maximization vs Supervised Fisher class-separation maximization.", "Project_Use": "Side-by-side 2D projections in Section 5."},
    {"Term": "mRMR", "Category": "Unit 2 (Dim. Reduction)", "Simple": "Picks features that strongly predict the target while avoiding duplicate clues.", "Technical": "Maximum Relevance Minimum Redundancy greedy F-statistic / correlation quotient.", "Project_Use": "Selects top 6 or 8 non-redundant indicators for prediction."},
    {"Term": "K-Means vs K-Medoids (PAM)", "Category": "Unit 3 (Clustering)", "Simple": "Both group similar stocks; K-Means uses an average center, while K-Medoids picks an actual real stock as the center.", "Technical": "Centroid-based Euclidean partitioning vs Medoid-based Manhattan partitioning.", "Project_Use": "Groups 12 NSE & NASDAQ stocks in Section 6."},
    {"Term": "Silhouette & Elbow Method", "Category": "Unit 3 (Clustering)", "Simple": "Charts that help us pick the best number of clusters (k).", "Technical": "WCSS inertia elbow curve and intra-cluster vs nearest-cluster separation score (-1 to +1).", "Project_Use": "Evaluates k=2..6 in Section 6."},
    {"Term": "K-Armed Bandit & Dyna-Q", "Category": "Unit 4 (RL)", "Simple": "Bandit tests which rule works best; Dyna-Q learns a mini-model of the market to practice extra steps.", "Technical": "Single-state epsilon-greedy bandit vs Model-Based RL combining direct TD updates with simulated planning.", "Project_Use": "Demonstrated in Section 7 RL Trading Simulation."},
    {"Term": "Binary vs Bipolar Sigmoid & ReLU", "Category": "Unit 5 (Neural Nets)", "Simple": "Switches that turn a neuron's score into a 0-to-1 probability, a -1-to-+1 signal, or a positive-only value.", "Technical": "Logistic sigmoid (0,1), hyperbolic tangent bipolar (-1,+1), and Rectified Linear Unit max(0,z).", "Project_Use": "Interactive Lab Exercise 9 calculator & MLP hidden layers in Section 8."},
    {"Term": "Backpropagation", "Category": "Unit 5 (Neural Nets)", "Simple": "How a neural network learns from its mistakes by sending error backward to adjust weights.", "Technical": "Chain-rule gradient computation from output loss backward through hidden layers.", "Project_Use": "Powers the MLP Classifier in Section 8 with live Training & Validation loss curves."},
]


# 13-Step Guided Learning Tour
GUIDED_TOUR_STEPS = [
    ("Step 1 – Understanding the Project", "STOCKMIND AI integrates all 5 units of MAI511-2 Advanced Machine Learning into one unified stock analytics platform.", "A real capstone must connect theory to an end-to-end working pipeline rather than isolated toy scripts.", "11-module Python architecture + dark glassmorphism Streamlit UI.", "Open Section 1 (Dashboard) to see KPIs, live charts, and model summaries."),
    ("Step 2 – Understanding the Stock Dataset", "Each trading day has Open, High, Low, Close, and Volume (OHLCV) records.", "Computers need clean numerical tables to learn patterns.", "12 Indian (NSE) and US (NASDAQ) stocks + custom ticker support.", "Inspect the latest closing price and dataset date range in Section 2."),
    ("Step 3 – Data Collection", "Fetches historical daily market data via Yahoo Finance (`yfinance`) with automatic local CSV caching.", "Ensures the app works reliably even if the API is temporarily slow or offline.", "`src/data_collection.py` with input validation and error handling.", "Check the active Data Source banner at the top of the page."),
    ("Step 4 – Data Preprocessing", "Sorts dates chronologically, removes duplicates, forward-fills missing values, and splits 80/20 by time.", "Random shuffling causes look-ahead bias (data leakage) in time series.", "`src/preprocessing.py` fitting `StandardScaler` strictly on `X_train`.", "View exact missing-value counts before and after cleaning in Section 2."),
    ("Step 5 – Feature Engineering", "Creates 19 technical indicators (SMA, EMA, RSI, MACD, Volatility, Volume Ratio, Lags) using only past/today data.", "Raw prices alone miss momentum and risk signals.", "`src/feature_engineering.py` with `Target_Next_Close = Close.shift(-1)`.", "Explore interactive Candlestick, MACD, and RSI charts in Section 3."),
    ("Step 6 – Regression Methods (Unit 1)", "Trains 10 regression models (OLS, GD, Poly, Ridge, Lasso, Trees, Model Tree) + Logistic Regression for direction.", "Compares linear, regularized, and nonlinear models and proves Overfitting vs Underfitting.", "`src/regression.py` evaluating MAE, MSE, RMSE, R², and directional Accuracy.", "Test Ridge/Lasso penalties interactively in Section 4."),
    ("Step 7 – Dimensionality Reduction (Unit 2)", "Applies Filters (Low Variance, High Corr, mRMR), Wrappers (SFS, SBS), and Projections (PCA, LDA, ICA, FA, MDS, Isomap).", "Removes redundant indicators and contrasts unsupervised PCA vs supervised LDA.", "`src/dimensionality_reduction.py` fit strictly on `X_train`.", "Compare Before-vs-After RMSE in Section 5."),
    ("Step 8 – Unsupervised Learning (Unit 3)", "Groups 12 stocks by risk-return behavior using K-Means, K-Medoids, Agglomerative, DIANA, BIRCH, and Chameleon.", "Discovers market regimes (Momentum vs Defensive) without labels.", "`src/clustering.py` with Hopkins Statistic, Elbow, Silhouette, and Dendrogram.", "Change k from 2 to 6 in Section 6 to watch clusters reorganize."),
    ("Step 9 – Reinforcement Learning (Unit 4)", "Simulates an agent choosing Buy/Sell/Hold across 18 discrete market states.", "Demonstrates K-Armed Bandit, Model-Free Q-Learning, and Model-Based Dyna-Q planning.", "`src/reinforcement_learning.py` (educational virtual-capital simulation).", "Compare RL Portfolio vs Buy-and-Hold in Section 7."),
    ("Step 10 – Neural Networks (Unit 5)", "Demonstrates single-neuron Sigmoid/ReLU math (Lab Ex 9) and trains an MLP classifier via Backpropagation (Lab Ex 10).", "Learns nonlinear Up/Down day patterns beyond single-layer perceptrons.", "`src/neural_network.py` tracking Training & Validation log-loss.", "Adjust neuron weights and MLP hidden layers in Section 8."),
    ("Step 11 – Stock Price Prediction", "Generates next-trading-day closing price estimates and 5-day statistical confidence bands.", "Combines preprocessing, feature selection, and trained models into a usable forecast.", "`src/prediction.py` supporting 7 selectable models.", "Compare Ridge vs Model Tree vs MLP forecasts in Section 9."),
    ("Step 12 – Model Evaluation", "Benchmarks all regression, classification, and feature-reduction models side by side.", "Proves which techniques generalize best on unseen chronological test data.", "Section 10 & Section 13 (Training & Evaluation Guide).", "Review Train vs Test R² gaps to explain model reliability."),
    ("Step 13 – Final Results & Viva", "Presents verified, non-fabricated metrics grounded in actual execution.", "Enables confident, transparent demonstration to faculty and peers.", "Complete codebase + README + live interactive presentation scripts.", "Switch to 'Faculty Presentation Mode' in the sidebar during your demo!"),
]
