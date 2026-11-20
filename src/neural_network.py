"""
Unit 5: Neural Networks & Multilayer Perceptron Module (src/neural_network.py)
Implements all MAI511-2 Unit 5 syllabus topics and Lab Exercises 9 & 10:
1. Biological Neuron vs Artificial Neuron comparison & ANN Evolution
2. Lab Exercise 9: Single Neuron Output Calculation using Binary Sigmoid, Bipolar Sigmoid, and ReLU
3. Single-Layer Perceptron (SLP) binary classifier
4. Lab Exercise 10: Multilayer Perceptron (MLP) with Backpropagation for Next-Day Price Direction Classification
5. Explicit Step-by-Step Backpropagation weight update demonstration
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    log_loss,
)


BIOLOGICAL_VS_ARTIFICIAL_TABLE = pd.DataFrame([
    {
        "Aspect": "Fundamental Processing Unit",
        "Biological Neural Network (BNN)": "Biological Neuron (Soma, Dendrites, Axon, Synapse)",
        "Artificial Neural Network (ANN)": "Artificial Node / Perceptron (Inputs, Weights, Bias, Activation)",
    },
    {
        "Aspect": "Input Signal Reception",
        "Biological Neural Network (BNN)": "Dendrites receive electrochemical impulses from connected neurons",
        "Artificial Neural Network (ANN)": "Input layer features x_1, x_2, ..., x_n (e.g., Close, RSI, MACD)",
    },
    {
        "Aspect": "Connection Strength",
        "Biological Neural Network (BNN)": "Synaptic strength (neurotransmitter release efficiency)",
        "Artificial Neural Network (ANN)": "Numerical synaptic weights w_1, w_2, ..., w_n learned via training",
    },
    {
        "Aspect": "Signal Integration",
        "Biological Neural Network (BNN)": "Soma (cell body) integrates incoming graded potentials",
        "Artificial Neural Network (ANN)": "Weighted summation net input: z = sum(w_i * x_i) + bias",
    },
    {
        "Aspect": "Firing / Output Mechanism",
        "Biological Neural Network (BNN)": "Axon fires an action potential when membrane threshold is exceeded",
        "Artificial Neural Network (ANN)": "Activation Function f(z) (Binary Sigmoid, Bipolar Sigmoid, ReLU)",
    },
    {
        "Aspect": "Learning Mechanism",
        "Biological Neural Network (BNN)": "Hebbian plasticity / Long-Term Potentiation (synaptic remodeling)",
        "Artificial Neural Network (ANN)": "Backpropagation of error using Gradient Descent weight updates",
    },
])


def binary_sigmoid(z):
    """Binary Sigmoid Activation Function (Range: 0 to 1): f(z) = 1 / (1 + e^(-z))."""
    z_clip = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_clip))


def bipolar_sigmoid(z):
    """Bipolar Sigmoid Activation Function (Range: -1 to +1): f(z) = (1 - e^(-z)) / (1 + e^(-z))."""
    z_clip = np.clip(z, -500, 500)
    return (1.0 - np.exp(-z_clip)) / (1.0 + np.exp(-z_clip))


def relu_activation(z):
    """Rectified Linear Unit (ReLU) Activation Function (Range: 0 to +inf): f(z) = max(0, z)."""
    return np.maximum(0.0, z)


def calculate_single_neuron_output(inputs: list, weights: list, bias: float):
    """
    Lab Exercise 9 Implementation:
    Calculates the net input z = sum(x_i * w_i) + b of a single artificial neuron
    and computes its output under Binary Sigmoid, Bipolar Sigmoid, ReLU, and Step activations.
    """
    x = np.array(inputs, dtype=float)
    w = np.array(weights, dtype=float)
    weighted_terms = x * w
    net_input_z = float(np.sum(weighted_terms) + bias)

    out_binary = float(binary_sigmoid(net_input_z))
    out_bipolar = float(bipolar_sigmoid(net_input_z))
    out_relu = float(relu_activation(net_input_z))
    out_step = 1 if net_input_z >= 0.0 else 0

    breakdown_df = pd.DataFrame({
        "Input_Index": [f"x_{i+1}" for i in range(len(x))],
        "Input_Value (x_i)": np.round(x, 4),
        "Synaptic_Weight (w_i)": np.round(w, 4),
        "Product (x_i * w_i)": np.round(weighted_terms, 4),
    })

    # Curve coordinates for plotting activation functions
    z_grid = np.linspace(-6.0, 6.0, 200)
    curves_df = pd.DataFrame({
        "z": z_grid,
        "Binary_Sigmoid": binary_sigmoid(z_grid),
        "Bipolar_Sigmoid": bipolar_sigmoid(z_grid),
        "ReLU": relu_activation(z_grid),
    })

    return {
        "net_input_z": round(net_input_z, 6),
        "bias": round(float(bias), 4),
        "binary_sigmoid_output": round(out_binary, 6),
        "bipolar_sigmoid_output": round(out_bipolar, 6),
        "relu_output": round(out_relu, 6),
        "step_perceptron_output": out_step,
        "direction_interpretation": "Bullish (Price Up)" if out_binary >= 0.5 else "Bearish (Price Down)",
        "breakdown_df": breakdown_df,
        "curves_df": curves_df,
    }


def run_mlp_and_perceptron_classification(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_dir_train: pd.Series,
    y_dir_test: pd.Series,
    hidden_layers: tuple = (32, 16),
    activation: str = "relu",
    learning_rate_init: float = 0.005,
    max_epochs: int = 80,
):
    """
    Lab Exercise 10 Implementation:
    Trains both a Single-Layer Perceptron (SLP) and a Multilayer Perceptron (MLP)
    using Backpropagation to classify whether the next trading day's closing price
    will be Higher (1) or Lower (0) than today's closing price.
    """
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    y_tr = y_dir_train.values.astype(int)
    y_te = y_dir_test.values.astype(int)

    # 1. Single-Layer Perceptron (Rosenblatt Linear Threshold Classifier)
    slp = Perceptron(max_iter=1000, eta0=0.01, random_state=42)
    slp.fit(X_tr_s, y_tr)
    slp_tr_acc = round(float(accuracy_score(y_tr, slp.predict(X_tr_s))) * 100.0, 2)
    slp_te_acc = round(float(accuracy_score(y_te, slp.predict(X_te_s))) * 100.0, 2)

    # 2. Multilayer Perceptron (MLP) trained epoch-by-epoch to record both Training & Validation loss curves
    # Hold out last 15% of chronological training set as validation monitor
    val_split = max(20, int(len(X_tr_s) * 0.85))
    X_sub_tr, X_val = X_tr_s[:val_split], X_tr_s[val_split:]
    y_sub_tr, y_val = y_tr[:val_split], y_tr[val_split:]

    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation=activation,
        solver="adam",
        alpha=0.01,
        learning_rate_init=learning_rate_init,
        max_iter=1,
        warm_start=True,
        random_state=42,
    )

    classes = np.array([0, 1])
    epoch_records = []
    for ep in range(1, max_epochs + 1):
        mlp.partial_fit(X_sub_tr, y_sub_tr, classes=classes)
        tr_prob = mlp.predict_proba(X_sub_tr)
        val_prob = mlp.predict_proba(X_val)
        tr_loss = float(log_loss(y_sub_tr, tr_prob, labels=classes))
        val_loss = float(log_loss(y_val, val_prob, labels=classes))
        epoch_records.append({
            "Epoch": ep,
            "Training_Loss": round(tr_loss, 4),
            "Validation_Loss": round(val_loss, 4),
        })

    # Final evaluation on full train and unseen chronological test set
    tr_pred = mlp.predict(X_tr_s)
    te_pred = mlp.predict(X_te_s)
    te_prob = mlp.predict_proba(X_te_s)[:, 1]

    cm = confusion_matrix(y_te, te_pred, labels=[0, 1])
    report_dict = classification_report(
        y_te, te_pred, labels=[0, 1], target_names=["Decrease (0)", "Increase (1)"], output_dict=True, zero_division=0
    )
    report_rows = []
    for label_name in ["Decrease (0)", "Increase (1)", "macro avg", "weighted avg"]:
        if label_name in report_dict:
            row = report_dict[label_name]
            report_rows.append({
                "Class / Metric": label_name,
                "Precision": round(float(row["precision"]), 4),
                "Recall": round(float(row["recall"]), 4),
                "F1_Score": round(float(row["f1-score"]), 4),
                "Support": int(row["support"]),
            })

    total_params = sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)
    architecture_summary = {
        "Input_Layer_Neurons": X_train.shape[1],
        "Hidden_Layers": str(hidden_layers),
        "Output_Layer_Neurons": 1,
        "Hidden_Activation": activation.upper(),
        "Output_Activation": "Binary Sigmoid (Logistic)",
        "Learning_Algorithm": "Backpropagation with Adam Gradient Descent",
        "Total_Trainable_Parameters": int(total_params),
    }

    return {
        "slp_train_accuracy": slp_tr_acc,
        "slp_test_accuracy": slp_te_acc,
        "mlp_train_accuracy": round(float(accuracy_score(y_tr, tr_pred)) * 100.0, 2),
        "mlp_test_accuracy": round(float(accuracy_score(y_te, te_pred)) * 100.0, 2),
        "mlp_precision": round(float(precision_score(y_te, te_pred, zero_division=0)) * 100.0, 2),
        "mlp_recall": round(float(recall_score(y_te, te_pred, zero_division=0)) * 100.0, 2),
        "mlp_f1": round(float(f1_score(y_te, te_pred, zero_division=0)) * 100.0, 2),
        "confusion_matrix": cm,
        "classification_report_df": pd.DataFrame(report_rows),
        "loss_history_df": pd.DataFrame(epoch_records),
        "architecture": architecture_summary,
        "mlp_model": mlp,
        "scaler": scaler,
        "test_probabilities": te_prob,
    }
