"""
ML-Based Timing Prediction for Digital Circuits
Author: Bhaskar Jha

Project:
This script uses Machine Learning to predict propagation delay of CMOS logic gates
using circuit-level parameters such as supply voltage, load capacitance, temperature,
fanout, transistor width, input transition time, and gate type.

Tools Used:
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Joblib

Outputs:
1. Synthetic timing dataset CSV
2. Predicted vs actual delay scatter plot
3. Accuracy improvement with training data plot
4. Feature importance graph
5. Model performance report
6. Trained ML model file
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -------------------------------------------------------------
# Create output folder
# -------------------------------------------------------------

OUTPUT_DIR = "ml_timing_prediction_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------------------------------------------------
# Step 1: Generate synthetic VLSI timing dataset
# -------------------------------------------------------------

def generate_timing_dataset(num_samples=3000, random_state=42):
    """
    Generates synthetic circuit timing data.

    In real projects, this data can come from:
    - LTSpice
    - Cadence Virtuoso
    - HSPICE
    - Ngspice
    - Synopsys PrimeTime reports

    For learning and resume demonstration, we create realistic synthetic data.
    """

    np.random.seed(random_state)

    gate_types = np.random.choice(
        ["INV", "NAND2", "NOR2", "XOR2", "AND2", "OR2"],
        size=num_samples
    )

    vdd = np.random.uniform(0.7, 1.2, num_samples)              # Supply voltage in volts
    load_cap = np.random.uniform(1, 50, num_samples)            # Load capacitance in fF
    temperature = np.random.uniform(25, 125, num_samples)       # Temperature in degree Celsius
    transistor_width = np.random.uniform(0.12, 2.0, num_samples) # Width in micrometer
    fanout = np.random.randint(1, 6, num_samples)               # Fanout count
    input_slew = np.random.uniform(5, 120, num_samples)         # Input transition time in ps

    gate_delay_factor = {
        "INV": 1.0,
        "NAND2": 1.35,
        "NOR2": 1.50,
        "XOR2": 1.80,
        "AND2": 1.45,
        "OR2": 1.55
    }

    delay = []

    for i in range(num_samples):
        gate_factor = gate_delay_factor[gate_types[i]]

        # Base timing model inspired by CMOS delay relationship:
        # Delay increases with load capacitance, temperature, fanout, and input slew.
        # Delay decreases with supply voltage and transistor width.
        base_delay = (
            gate_factor
            * (load_cap[i] * 1.8)
            * (1 / vdd[i])
            * (1 / np.sqrt(transistor_width[i]))
        )

        temp_effect = 1 + 0.003 * (temperature[i] - 25)
        fanout_effect = 1 + 0.12 * fanout[i]
        slew_effect = 1 + 0.004 * input_slew[i]

        noise = np.random.normal(0, 4)

        final_delay = base_delay * temp_effect * fanout_effect * slew_effect + noise

        delay.append(max(final_delay, 1))

    data = pd.DataFrame({
        "Gate_Type": gate_types,
        "VDD_V": vdd,
        "Load_Cap_fF": load_cap,
        "Temperature_C": temperature,
        "Transistor_Width_um": transistor_width,
        "Fanout": fanout,
        "Input_Slew_ps": input_slew,
        "Propagation_Delay_ps": delay
    })

    return data


# -------------------------------------------------------------
# Step 2: Train and evaluate ML models
# -------------------------------------------------------------

def train_models(data):
    """
    Trains multiple ML models and compares their performance.
    """

    X = data.drop("Propagation_Delay_ps", axis=1)
    y = data["Propagation_Delay_ps"]

    categorical_features = ["Gate_Type"]
    numerical_features = [
        "VDD_V",
        "Load_Cap_fF",
        "Temperature_C",
        "Transistor_Width_um",
        "Fanout",
        "Input_Slew_ps"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", StandardScaler(), numerical_features)
        ]
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        )
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    results = []
    trained_pipelines = {}

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results.append({
            "Model": model_name,
            "MAE_ps": mae,
            "RMSE_ps": rmse,
            "R2_Score": r2
        })

        trained_pipelines[model_name] = {
            "pipeline": pipeline,
            "y_test": y_test,
            "y_pred": y_pred,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test_full": y_test
        }

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="R2_Score", ascending=False)

    best_model_name = results_df.iloc[0]["Model"]
    best_pipeline_data = trained_pipelines[best_model_name]

    return results_df, best_model_name, best_pipeline_data, X, y


# -------------------------------------------------------------
# Step 3: Plot predicted vs actual delay
# -------------------------------------------------------------

def plot_predicted_vs_actual(y_test, y_pred, best_model_name):
    plt.figure(figsize=(8, 6))

    plt.scatter(y_test, y_pred, alpha=0.7, edgecolors="black")

    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        linewidth=2
    )

    plt.xlabel("Actual Propagation Delay (ps)")
    plt.ylabel("Predicted Propagation Delay (ps)")
    plt.title(f"Predicted vs Actual Delay - {best_model_name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "predicted_vs_actual_delay.png")
    plt.savefig(path, dpi=300)
    plt.show()


# -------------------------------------------------------------
# Step 4: Plot accuracy improvement with training data
# -------------------------------------------------------------

def plot_learning_curve(best_pipeline, X, y):
    train_sizes, train_scores, test_scores = learning_curve(
        best_pipeline,
        X,
        y,
        cv=5,
        scoring="r2",
        train_sizes=np.linspace(0.1, 1.0, 8),
        n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(8, 6))

    plt.plot(train_sizes, train_mean, marker="o", label="Training Accuracy")
    plt.plot(train_sizes, test_mean, marker="s", label="Validation Accuracy")

    plt.xlabel("Number of Training Samples")
    plt.ylabel("R² Score")
    plt.title("Accuracy Improvement with More Training Data")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "accuracy_improvement_training_data.png")
    plt.savefig(path, dpi=300)
    plt.show()


# -------------------------------------------------------------
# Step 5: Plot feature importance
# -------------------------------------------------------------

def plot_feature_importance(best_pipeline, best_model_name):
    """
    Feature importance works directly for tree-based models.
    """

    model = best_pipeline.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        print("Feature importance is not available for this model.")
        return

    preprocessor = best_pipeline.named_steps["preprocessor"]

    cat_features = preprocessor.named_transformers_["cat"].get_feature_names_out(["Gate_Type"])
    num_features = np.array([
        "VDD_V",
        "Load_Cap_fF",
        "Temperature_C",
        "Transistor_Width_um",
        "Fanout",
        "Input_Slew_ps"
    ])

    all_features = np.concatenate([cat_features, num_features])
    importances = model.feature_importances_

    feature_df = pd.DataFrame({
        "Feature": all_features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_df["Feature"], feature_df["Importance"])
    plt.xlabel("Importance Score")
    plt.ylabel("Circuit Feature")
    plt.title(f"Feature Importance - {best_model_name}")
    plt.gca().invert_yaxis()
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(path, dpi=300)
    plt.show()

    feature_df.to_csv(
        os.path.join(OUTPUT_DIR, "feature_importance.csv"),
        index=False
    )


# -------------------------------------------------------------
# Step 6: Predict delay for a new circuit
# -------------------------------------------------------------

def predict_new_circuit(best_pipeline):
    """
    Example prediction for a new circuit condition.
    """

    new_circuit = pd.DataFrame({
        "Gate_Type": ["NAND2"],
        "VDD_V": [1.0],
        "Load_Cap_fF": [20],
        "Temperature_C": [75],
        "Transistor_Width_um": [0.8],
        "Fanout": [3],
        "Input_Slew_ps": [50]
    })

    predicted_delay = best_pipeline.predict(new_circuit)[0]

    print("\nNew Circuit Timing Prediction")
    print("-----------------------------")
    print(new_circuit)
    print(f"\nPredicted Propagation Delay: {predicted_delay:.2f} ps")


# -------------------------------------------------------------
# Step 7: Generate model report
# -------------------------------------------------------------

def generate_report(results_df, best_model_name):
    report_path = os.path.join(OUTPUT_DIR, "model_performance_report.txt")

    with open(report_path, "w") as file:
        file.write("ML-Based Timing Prediction Report\n")
        file.write("=================================\n\n")

        file.write("Project Objective:\n")
        file.write(
            "To predict propagation delay of digital CMOS circuits using machine learning "
            "without performing full circuit simulation for every new condition.\n\n"
        )

        file.write("Input Features Used:\n")
        file.write("- Gate Type\n")
        file.write("- Supply Voltage VDD\n")
        file.write("- Load Capacitance\n")
        file.write("- Temperature\n")
        file.write("- Transistor Width\n")
        file.write("- Fanout\n")
        file.write("- Input Slew\n\n")

        file.write("Model Comparison:\n")
        file.write(results_df.to_string(index=False))
        file.write("\n\n")

        file.write(f"Best Performing Model: {best_model_name}\n\n")

        file.write("Generated Outputs:\n")
        file.write("1. predicted_vs_actual_delay.png\n")
        file.write("2. accuracy_improvement_training_data.png\n")
        file.write("3. feature_importance.png\n")
        file.write("4. feature_importance.csv\n")
        file.write("5. trained_timing_prediction_model.pkl\n")
        file.write("6. timing_dataset.csv\n\n")

        file.write("Resume Value:\n")
        file.write(
            "This project demonstrates the application of AI/ML in VLSI timing analysis, "
            "reducing dependency on repeated full simulations and enabling faster delay prediction.\n"
        )

    print(f"\nReport saved at: {report_path}")


# -------------------------------------------------------------
# Main function
# -------------------------------------------------------------

def main():
    print("ML-Based Timing Prediction for Digital Circuits")
    print("------------------------------------------------")

    print("\nGenerating timing dataset...")
    data = generate_timing_dataset(num_samples=3000)

    dataset_path = os.path.join(OUTPUT_DIR, "timing_dataset.csv")
    data.to_csv(dataset_path, index=False)

    print(f"Dataset saved at: {dataset_path}")
    print("\nSample Dataset:")
    print(data.head())

    print("\nTraining ML models...")
    results_df, best_model_name, best_pipeline_data, X, y = train_models(data)

    print("\nModel Performance Comparison:")
    print(results_df)

    best_pipeline = best_pipeline_data["pipeline"]
    y_test = best_pipeline_data["y_test"]
    y_pred = best_pipeline_data["y_pred"]

    print(f"\nBest Model Selected: {best_model_name}")

    print("\nGenerating predicted vs actual delay graph...")
    plot_predicted_vs_actual(y_test, y_pred, best_model_name)

    print("\nGenerating learning curve...")
    plot_learning_curve(best_pipeline, X, y)

    print("\nGenerating feature importance graph...")
    plot_feature_importance(best_pipeline, best_model_name)

    print("\nPredicting delay for a new circuit...")
    predict_new_circuit(best_pipeline)

    model_path = os.path.join(OUTPUT_DIR, "trained_timing_prediction_model.pkl")
    joblib.dump(best_pipeline, model_path)

    print(f"\nTrained model saved at: {model_path}")

    results_df.to_csv(
        os.path.join(OUTPUT_DIR, "model_comparison_results.csv"),
        index=False
    )

    generate_report(results_df, best_model_name)

    print("\nProject completed successfully.")
    print(f"All outputs are saved inside: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
