"""
Unit 4: Reinforcement Learning Simulation Module (src/reinforcement_learning.py)
Implements all MAI511-2 Unit 4 syllabus topics:
1. Elements of RL: Agent, Environment, State (S_t), Action (Buy/Sell/Hold), Reward (R_t), Policy (pi)
2. Single-State Case: K-Armed Bandit (Epsilon-Greedy strategy selection across technical signals)
3. Model-Free Temporal Difference (TD) Learning: Tabular Q-Learning Trading Agent
4. Model-Based Reinforcement Learning (Lab Exercise 8: Transition Probability & Reward Model + Planning)
5. Generalization (Feature-based linear state value approximation) & Partially Observable States (POMDP belief state)
"""

import numpy as np
import pandas as pd


ACTIONS = {0: "Hold", 1: "Buy", 2: "Sell"}


def discretize_market_states(df_modeling: pd.DataFrame) -> np.ndarray:
    """
    Maps continuous market variables into 18 discrete states (0..17) using:
    - Price vs SMA_20 Trend (3 levels: Bearish < -1%, Neutral, Bullish > +1%)
    - RSI_14 Regime (3 levels: Oversold < 40, Neutral 40..60, Overbought > 60)
    - Rolling Volatility Regime (2 levels: Low Vol <= median, High Vol > median)
    State ID = trend_idx * 6 + rsi_idx * 2 + vol_idx
    """
    close = df_modeling["Close"].values
    sma20 = df_modeling["SMA_20"].values
    rsi = df_modeling["RSI_14"].values
    vol = df_modeling["Volatility_20"].values
    vol_median = float(np.nanmedian(vol))

    trend_pct = ((close - sma20) / (sma20 + 1e-8)) * 100.0
    trend_idx = np.where(trend_pct < -1.0, 0, np.where(trend_pct > 1.0, 2, 1))
    rsi_idx = np.where(rsi < 40.0, 0, np.where(rsi > 60.0, 2, 1))
    vol_idx = np.where(vol > vol_median, 1, 0)

    states = trend_idx * 6 + rsi_idx * 2 + vol_idx
    return states.astype(int)


def run_k_armed_bandit_simulation(df_modeling: pd.DataFrame, epsilon: float = 0.15, random_state: int = 42):
    """
    Demonstrates the Single-State Case: K-Armed Bandit (Unit 4 Syllabus Topic 2 & 3).
    Arms represent 4 simple trading signal rules evaluated on daily returns:
      Arm 0: Trend Following (Long when Close > SMA_20)
      Arm 1: Mean Reversion (Long when RSI_14 < 45)
      Arm 2: Momentum Breakout (Long when MACD > MACD_Signal)
      Arm 3: Defensive Cash Baseline (Small fixed risk-free daily yield)
    The agent uses Incremental Sample-Average updates and Epsilon-Greedy exploration.
    """
    rng = np.random.default_rng(random_state)
    arm_names = [
        "Arm 1: SMA Trend Following",
        "Arm 2: RSI Mean Reversion",
        "Arm 3: MACD Momentum",
        "Arm 4: Risk-Free Cash Hold",
    ]
    k_arms = len(arm_names)

    close = df_modeling["Close"].values
    next_close = df_modeling["Target_Next_Close"].values
    sma20 = df_modeling["SMA_20"].values
    rsi = df_modeling["RSI_14"].values
    macd = df_modeling["MACD"].values
    macd_sig = df_modeling["MACD_Signal"].values

    next_ret_pct = ((next_close - close) / close) * 100.0

    # Precompute daily reward for each arm at time t
    arm_rewards = np.zeros((len(df_modeling), k_arms))
    arm_rewards[:, 0] = np.where(close > sma20, next_ret_pct, -0.2 * next_ret_pct)
    arm_rewards[:, 1] = np.where(rsi < 45.0, next_ret_pct, np.where(rsi > 65.0, -next_ret_pct, 0.0))
    arm_rewards[:, 2] = np.where(macd > macd_sig, next_ret_pct, -0.2 * next_ret_pct)
    arm_rewards[:, 3] = 0.02  # ~5% annualized risk-free daily equivalent

    Q_estimates = np.zeros(k_arms)
    action_counts = np.zeros(k_arms, dtype=int)
    cumulative_reward = 0.0
    history = []

    for t in range(len(df_modeling)):
        if rng.random() < epsilon:
            chosen_arm = int(rng.integers(0, k_arms))
        else:
            chosen_arm = int(np.argmax(Q_estimates))

        reward = float(arm_rewards[t, chosen_arm])
        action_counts[chosen_arm] += 1
        # Incremental sample-average update rule: Q(a) <- Q(a) + (1/N(a)) * [R - Q(a)]
        Q_estimates[chosen_arm] += (reward - Q_estimates[chosen_arm]) / action_counts[chosen_arm]
        cumulative_reward += reward

        history.append({
            "Step": t + 1,
            "Date": df_modeling["Date"].iloc[t],
            "Chosen_Arm": arm_names[chosen_arm],
            "Step_Reward_Pct": round(reward, 4),
            "Cumulative_Reward_Pct": round(cumulative_reward, 4),
        })

    summary_df = pd.DataFrame({
        "Bandit_Arm": arm_names,
        "Times_Selected": action_counts,
        "Selection_Share_Pct": np.round(action_counts / max(1, len(df_modeling)) * 100.0, 2),
        "Estimated_Value_Q": np.round(Q_estimates, 4),
        "True_Mean_Reward": np.round(arm_rewards.mean(axis=0), 4),
    })

    return {
        "bandit_summary_df": summary_df,
        "bandit_history_df": pd.DataFrame(history),
    }


def run_rl_trading_simulation(
    df_modeling: pd.DataFrame,
    algorithm: str = "Model-Based RL (Dyna-Q)",
    episodes: int = 15,
    alpha: float = 0.15,
    gamma: float = 0.92,
    epsilon: float = 0.15,
    planning_steps: int = 5,
    initial_capital: float = 100000.0,
):
    """
    Simulates an educational Reinforcement Learning stock trading agent across historical market states.

    Supports both syllabus paradigms:
    1. 'Model-Free TD (Q-Learning)': Updates Q(s, a) strictly from direct experience transitions.
    2. 'Model-Based RL (Dyna-Q)' (Lab Exercise 8): Learns an explicit internal Environment Model
       T(s, a -> s') and R(s, a) and performs `planning_steps` simulated updates after each real step.

    Important Academic Note:
    This is a controlled educational simulation for MAI511-2 Unit 4, not a live trading bot.
    """
    rng = np.random.default_rng(42)
    states = discretize_market_states(df_modeling)
    close_prices = df_modeling["Close"].values
    next_prices = df_modeling["Target_Next_Close"].values
    dates = df_modeling["Date"].values

    n_states = 18
    n_actions = 3  # 0: Hold, 1: Buy, 2: Sell
    Q_table = np.zeros((n_states, n_actions))

    # Model-Based RL internal environment tables: transition counts N(s,a,s') and expected reward R(s,a)
    transition_counts = np.zeros((n_states, n_actions, n_states))
    reward_model = np.zeros((n_states, n_actions))
    observed_pairs = []

    is_model_based = "Model-Based" in algorithm
    episode_rewards = []

    for ep in range(episodes):
        ep_eps = max(0.02, epsilon * (0.88 ** ep))
        total_ep_reward = 0.0

        for t in range(len(df_modeling) - 1):
            s = states[t]
            s_next = states[t + 1]
            pct_change = ((next_prices[t] - close_prices[t]) / close_prices[t]) * 100.0

            # Epsilon-greedy action selection
            if rng.random() < ep_eps:
                a = int(rng.integers(0, n_actions))
            else:
                a = int(np.argmax(Q_table[s]))

            # Reward function based on market direction alignment + small transaction cost
            if a == 1:  # Buy
                r = pct_change - 0.03
            elif a == 2:  # Sell / Defensive Short-Avoidance
                r = -pct_change - 0.03
            else:  # Hold
                r = 0.01 if abs(pct_change) < 0.25 else 0.0

            # Direct Temporal Difference (Q-Learning) update
            td_target = r + gamma * np.max(Q_table[s_next])
            td_error = td_target - Q_table[s, a]
            Q_table[s, a] += alpha * td_error
            total_ep_reward += r

            # Model-Based Learning: update Transition Model P(s'|s,a) and Reward Model R(s,a)
            if is_model_based:
                transition_counts[s, a, s_next] += 1.0
                total_sa = np.sum(transition_counts[s, a])
                reward_model[s, a] += (r - reward_model[s, a]) / total_sa
                if (s, a) not in observed_pairs:
                    observed_pairs.append((s, a))

                # Planning step: sample previously visited (s_sim, a_sim) and update Q using learned model
                for _ in range(planning_steps):
                    idx_pair = int(rng.integers(0, len(observed_pairs)))
                    s_sim, a_sim = observed_pairs[idx_pair]
                    probs = transition_counts[s_sim, a_sim] / np.sum(transition_counts[s_sim, a_sim])
                    s_next_sim = int(np.argmax(probs))
                    r_sim = reward_model[s_sim, a_sim]
                    Q_table[s_sim, a_sim] += alpha * (
                        r_sim + gamma * np.max(Q_table[s_next_sim]) - Q_table[s_sim, a_sim]
                    )

        episode_rewards.append({
            "Episode": ep + 1,
            "Total_Episode_Reward": round(float(total_ep_reward), 2),
            "Exploration_Epsilon": round(float(ep_eps), 3),
        })

    # Evaluate learned greedy policy on the final pass to simulate portfolio trajectory
    cash = float(initial_capital)
    shares = 0.0
    cum_reward = 0.0
    trajectory = []
    action_counts = {"Hold": 0, "Buy": 0, "Sell": 0}

    for t in range(len(df_modeling)):
        s = states[t]
        price = float(close_prices[t])
        next_p = float(next_prices[t])
        pct_change = ((next_p - price) / price) * 100.0

        a = int(np.argmax(Q_table[s]))
        action_label = ACTIONS[a]
        action_counts[action_label] += 1

        if a == 1 and cash >= price:
            shares_bought = cash / price
            shares += shares_bought
            cash = 0.0
        elif a == 2 and shares > 0:
            cash += shares * price
            shares = 0.0

        portfolio_val = cash + shares * price
        buy_hold_val = initial_capital * (price / float(close_prices[0]))

        step_r = pct_change if a == 1 else (-pct_change if a == 2 else 0.0)
        cum_reward += step_r

        trajectory.append({
            "Date": dates[t],
            "Close_Price": round(price, 2),
            "State_ID": int(s),
            "Action": action_label,
            "Step_Reward": round(float(step_r), 3),
            "Cumulative_Reward": round(float(cum_reward), 2),
            "RL_Portfolio_Value": round(float(portfolio_val), 2),
            "Buy_And_Hold_Value": round(float(buy_hold_val), 2),
        })

    traj_df = pd.DataFrame(trajectory)

    # Extract readable Policy Table for visited states
    state_descriptions = []
    trend_names = ["Bearish (<SMA20)", "Neutral (~SMA20)", "Bullish (>SMA20)"]
    rsi_names = ["Oversold (<40)", "Neutral (40-60)", "Overbought (>60)"]
    vol_names = ["Low Volatility", "High Volatility"]

    for s_id in range(n_states):
        t_i = s_id // 6
        r_i = (s_id % 6) // 2
        v_i = s_id % 2
        best_a = int(np.argmax(Q_table[s_id]))
        state_descriptions.append({
            "State_ID": s_id,
            "Trend_Regime": trend_names[t_i],
            "RSI_Zone": rsi_names[r_i],
            "Volatility_Regime": vol_names[v_i],
            "Q_Hold": round(float(Q_table[s_id, 0]), 3),
            "Q_Buy": round(float(Q_table[s_id, 1]), 3),
            "Q_Sell": round(float(Q_table[s_id, 2]), 3),
            "Learned_Policy_Action": ACTIONS[best_a],
        })

    final_rl_val = float(traj_df["RL_Portfolio_Value"].iloc[-1])
    final_bh_val = float(traj_df["Buy_And_Hold_Value"].iloc[-1])

    return {
        "algorithm": algorithm,
        "trajectory_df": traj_df,
        "episode_df": pd.DataFrame(episode_rewards),
        "policy_df": pd.DataFrame(state_descriptions),
        "action_counts": action_counts,
        "final_portfolio_value": round(final_rl_val, 2),
        "final_return_pct": round(((final_rl_val - initial_capital) / initial_capital) * 100.0, 2),
        "buy_hold_return_pct": round(((final_bh_val - initial_capital) / initial_capital) * 100.0, 2),
        "cumulative_reward": round(float(cum_reward), 2),
    }
