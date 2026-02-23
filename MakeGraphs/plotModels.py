import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from Utils.loadData import load_data, prepare_data
from Utils.crossValidation import refit_models


#######################################################################################################
# REFIT_AND_PLOT_UNIVARIATE — Refit and visualize univariate models using best CVRuns parameters
# ----------------------------------------------------------------------------------------------------
#    Parameters:
#    ---------------------------
#    feature : str
#        The univariate input feature to visualize (e.g., "P_area").
#
#    Returns:
#    ---------------------------
#    None
#######################################################################################################

def refit_and_plot_univariate(feature="P_area", folder="CVRuns"):

    # ----------------------------------------------------------------------------
    # Load data
    # ----------------------------------------------------------------------------
    df_train, df_test = load_data("Data/dataset.xlsx")
    features = [feature]

    # ----------------------------------------------------------------------------
    # Find best hyperparameter CSV in CVRuns
    # ----------------------------------------------------------------------------
    pattern = re.compile(r"features-\[(.*?)\]\.csv")
    best_csv = None
    for f in os.listdir(folder):
        match = pattern.match(f)
        if match:
            feature_list = [x.strip() for x in match.group(1).split(",")]
            if sorted(feature_list) == sorted(features):
                best_csv = os.path.join(folder, f)
                break

    if best_csv is None:
        raise FileNotFoundError(f"No CVRuns file found for features={features}")

    print(f"Using best hyperparameters from: {best_csv}")

    summary_df = pd.read_csv(best_csv)

    # ----------------------------------------------------------------------------
    # Refit the models with best hyperparameters
    # ----------------------------------------------------------------------------
    final_summary, trained_models, (X_train, y_train, X_test, y_test) = refit_models(
        df_train, df_test, summary_df, features=features
    )

    # ----------------------------------------------------------------------------
    # Visualization setup
    # ----------------------------------------------------------------------------

    # Ensure pandas structures
    X_train = pd.DataFrame(X_train)
    X_test  = pd.DataFrame(X_test)
    y_train = pd.Series(y_train)
    y_test  = pd.Series(y_test)

    # Select only known models
    selected_models = ["Linear Regression", "Group-Wise Linear", "KNN", "Decision Tree", "SVR", "MLP"]
    trained_models = {k: v for k, v in trained_models.items() if k in selected_models}

    # Define common x-grid for smooth regression curve
    x_min, x_max = X_train[feature].min(), X_train[feature].max()
    x_grid = np.linspace(x_min, x_max, 300)
    X_grid = pd.DataFrame({feature: x_grid})

    # ----------------------------------------------------------------------------
    # Loop through models and plot
    # ----------------------------------------------------------------------------
    for model_name, model in trained_models.items():
        y_pred_train = model.predict(X_train)
        y_pred_test  = model.predict(X_test)
        y_pred_curve = model.predict(X_grid)

        # Compute log-error for visualization
        err_train = np.log10(10 ** y_pred_train / 10 ** y_train)
        err_test  = np.log10(10 ** y_pred_test / 10 ** y_test)

        # Plot: regression + residuals
        fig, (ax_main, ax_err) = plt.subplots(2, 1, figsize=(6.5, 6.5),
                                              gridspec_kw={"height_ratios": [3, 1]})

        # Top: regression curve
        ax_main.scatter(10 ** X_train[feature], 10 ** y_train, color="steelblue", alpha=0.6, label="Train")
        ax_main.scatter(10 ** X_test[feature], 10 ** y_test, color="orange", alpha=0.7, label="Test")
        ax_main.plot(10 ** x_grid, 10 ** y_pred_curve, color="crimson", lw=2.0, label="Model prediction")
        ax_main.set_xscale("log")
        ax_main.set_yscale("log")
        ax_main.set_xlabel(f"log₁₀({feature})", fontsize=12)
        ax_main.set_ylabel("log₁₀(Die area)", fontsize=12)
        ax_main.set_title(model_name, fontsize=13)
        ax_main.legend(fontsize=9)
        ax_main.grid(True, which="both", linestyle="--", alpha=0.4)

        # Bottom: residuals (log error)
        ax_err.scatter(10**y_train, err_train, color="steelblue", alpha=0.6)
        ax_err.scatter(10 **y_test, err_test, color="orange", alpha=0.7)
        ax_err.set_xscale("log")
        ax_err.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax_err.set_xlabel("log₁₀(Die area)", fontsize=12)
        ax_err.set_ylabel("log₁₀(pred/true)", fontsize=11)
        ax_err.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        plt.show()


#######################################################################################################
# PLOT_GROUPWISE_LINEAR_MODELS — Visualize group-wise linear regressions by package type
# ----------------------------------------------------------------------------------------------------
#    Parameters:
#    ---------------------------
#    features : list of str
#        The list of features, typically ["P_area", "P_type"].
#
#    Returns:
#    ---------------------------
#    None
#        Generates a panel of plots (one per package type) and prints linear equations.
#######################################################################################################

def plot_groupwise_linear_models(features=["P_area", "P_type"], folder="CVRuns"):
    ###################################################################################################
    # PART 1 - LOAD AND PREPARE DATA
    ###################################################################################################
    df_train, df_test = load_data("Data/dataset.xlsx")
    X_train, y_train, X_test, y_test, _ = prepare_data(df_train, features, df_test=df_test)

    if "P_type" not in X_train.columns:
        raise ValueError("Dataset must contain a 'P_type' column for group-wise regression.")

    ###################################################################################################
    # PART 2 - FIND BEST HYPERPARAMETER CSV
    ###################################################################################################
    pattern = re.compile(r"features-\[(.*?)\]\.csv")
    best_csv = None
    for f in os.listdir(folder):
        match = pattern.match(f)
        if match:
            feature_list = [x.strip() for x in match.group(1).split(",")]
            if sorted(feature_list) == sorted(features):
                best_csv = os.path.join(folder, f)
                break

    if best_csv is None:
        raise FileNotFoundError(f"No CVRuns file found for features={features}")

    ###################################################################################################
    # PART 3 - TRAIN SEPARATE MODELS FOR EACH PACKAGE TYPE
    ###################################################################################################
    numeric_feature = [f for f in features if f != "P_type"][0]
    group_feature = "P_type"
    groups = X_train[group_feature].unique()
    n_groups = len(groups)

    fig, axes = plt.subplots(n_groups, 2, figsize=(12, 3.5 * n_groups),
                             gridspec_kw={"height_ratios": [1]*n_groups})
    if n_groups == 1:
        axes = np.array([axes])

    for i, group_value in enumerate(groups):
        train_mask = X_train[group_feature] == group_value
        test_mask  = X_test[group_feature] == group_value

        X_train_g = X_train.loc[train_mask, [numeric_feature]]
        X_test_g  = X_test.loc[test_mask, [numeric_feature]]
        y_train_g = y_train.loc[train_mask]
        y_test_g  = y_test.loc[test_mask]

        # Fit linear regression for this group
        model = LinearRegression().fit(X_train_g, y_train_g)

        # Prediction curve
        x_grid = np.linspace(X_train_g[numeric_feature].min(), X_train_g[numeric_feature].max(), 300)
        y_grid = model.predict(x_grid.reshape(-1, 1))

        ###################################################################################################
        # PART 4 - PLOTS
        ###################################################################################################
        ax_main = axes[i, 0]
        ax_err  = axes[i, 1]

        # Main regression plot
        ax_main.scatter(10 ** X_train_g[numeric_feature], 10 ** y_train_g,
                        color="steelblue", alpha=0.6, label="Train")
        ax_main.scatter(10 ** X_test_g[numeric_feature], 10 ** y_test_g,
                        color="orange", alpha=0.7, label="Test")
        ax_main.plot(10 ** x_grid, 10 ** y_grid, color="crimson", lw=2.0, label="Model prediction")
        ax_main.set_xscale("log")
        ax_main.set_yscale("log")
        ax_main.set_title(group_value, fontsize=12, pad=10)
        ax_main.set_xlabel(f"log₁₀({numeric_feature})", fontsize=12)
        ax_main.set_ylabel("log₁₀(Die area)", fontsize=12)
        ax_main.grid(True, which="both", linestyle="--", alpha=0.4)
        ax_main.legend(fontsize=9, loc="best")

        # Residual plot
        y_pred_train = model.predict(X_train_g)
        y_pred_test  = model.predict(X_test_g)
        err_train = np.log10(10 ** y_pred_train / 10 ** y_train_g)
        err_test  = np.log10(10 ** y_pred_test / 10 ** y_test_g)

        ax_err.scatter(10 ** y_train_g, err_train, color="steelblue", alpha=0.6)
        ax_err.scatter(10 ** y_test_g, err_test, color="orange", alpha=0.7)
        ax_err.set_xscale("log")
        ax_err.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax_err.set_xlabel("log₁₀(Die area)", fontsize=11)
        ax_err.set_ylabel("Error log₁₀(pred/true)", fontsize=11)
        ax_err.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()