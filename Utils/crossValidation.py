import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, LeaveOneOut, cross_validate
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, KFold
from math import sqrt


from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings("ignore")

from Utils.loadData import prepare_data
from Utils.groupWiseLinear import GroupWiseLinearRegressor



#######################################################################################################
# OUR MODELS 
#######################################################################################################

def make_model_grid(preprocessor, features):

    grid = {
        "Linear Regression": {
            "model": Pipeline([
                ("preprocessor", preprocessor),
                ("regressor", LinearRegression())
            ]),
            "params": {}
        },
        "KNN": {
            "model": Pipeline([
                ("preprocessor", preprocessor),
                ("regressor", KNeighborsRegressor())
            ]),
            "params": {
                "regressor__n_neighbors": [1, 3, 5, 7, 10],
                "regressor__weights": ["uniform", "distance"],
                "regressor__p": [1, 2]
            }
        },
        "Decision Tree": {
            "model": Pipeline([
                ("preprocessor", preprocessor),
                ("regressor", DecisionTreeRegressor(random_state=0))
            ]),
            "params": {
                "regressor__max_depth": [2, 3, 5, 10, None],
                "regressor__min_samples_split": [2, 5, 10],
                "regressor__min_samples_leaf": [1, 2, 5]
            }
        },
        "SVR": {
            "model": Pipeline([
                ("preprocessor", preprocessor),
                ("regressor", SVR())
            ]),
            "params": {
                "regressor__C": [0.1, 1.0, 10],
                "regressor__epsilon": [0.05, 0.1, 0.2],
                "regressor__kernel": ["linear", "rbf", "poly"],
                "regressor__degree": [2, 3]
            }
        },
        "MLP": {
            "model": Pipeline([
                ("preprocessor", preprocessor),
                ("regressor", MLPRegressor(random_state=0, max_iter=2000))
            ]),
            "params": {
                "regressor__hidden_layer_sizes": [(20,), (50,), (100,), (50, 20)],
                "regressor__activation": ["relu", "tanh"],
                "regressor__alpha": [0.0001, 0.001, 0.01],
                "regressor__learning_rate_init": [0.001, 0.01]
            }
        }
    }

    # Add Group-Wise Linear only if 'P_type' is part of features
    if "P_type" in features:
        grid["Group-Wise Linear"] = {
            "model": GroupWiseLinearRegressor(),
            "params": {}
        }

    return grid


#######################################################################################################
# TRAIN AND EVALUATE REGRESSION MODELS WITH GRID SEARCH (CROSS-VALIDATED)
# ----------------------------------------------------------------------------------------------------
#    Performs hyperparameter tuning via GridSearchCV and evaluates model performance using
#    Leave-One-Out Cross-Validation (LOOCV). The Root Mean Squared Error (RMSE) is used
#    as the sole evaluation metric.
#
#    Parameters:
#    ---------------------------
#    df_train : pd.DataFrame
#        Training dataframe.
#
#    features : list of str
#        List of features to use for training.
#
#    Returns:
#    ---------------------------
#    summary_df : pd.DataFrame
#        DataFrame containing:
#        - Model name
#        - Best hyperparameters (from GridSearchCV minimizing RMSE)
#        - Mean ± std of RMSE from LOOCV
#######################################################################################################

def evaluate_models_with_grid_search(df_train, features=['M']):

    # ------------------------------------------------------------------------------------------------
    # Prepare training data 
    # ------------------------------------------------------------------------------------------------
    X, y, preprocessor = prepare_data(
        df_train,
        features=features,
        df_test=None,
    )

    # ------------------------------------------------------------------------------------------------
    # Define model grid
    # ------------------------------------------------------------------------------------------------
    model_grid   = make_model_grid(preprocessor, features)
    summary_rows = []

    # ------------------------------------------------------------------------------------------------
    # Cross-validation strategy for final evaluation
    # ------------------------------------------------------------------------------------------------
    #cv_eval = LeaveOneOut()
    cv_eval = KFold(n_splits=15, shuffle=False)

    # ------------------------------------------------------------------------------------------------
    # Loop over all models
    # ------------------------------------------------------------------------------------------------
    for name, config in model_grid.items():
        print(name)

        # --------------------------------------------------------------------------------------------
        # Step 1: Hyperparameter selection
        # --------------------------------------------------------------------------------------------
        grid = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            scoring="neg_root_mean_squared_error",
            cv=KFold(n_splits=5, shuffle=True, random_state=0), 
            n_jobs=-1,
            return_train_score=True
        )
        grid.fit(X, y)

        best_model = grid.best_estimator_

        # --------------------------------------------------------------------------------------------
        # Step 2: Train / validation performance with CV
        # --------------------------------------------------------------------------------------------
        cv_scores = cross_validate(
            best_model,
            X,
            y,
            cv=cv_eval,
            scoring="neg_root_mean_squared_error",
            return_train_score=True,
            n_jobs=-1
        )

        train_rmse = -cv_scores["train_score"]
        val_rmse   = -cv_scores["test_score"]

        summary_rows.append({
            "Model": name,
            "Best Params": grid.best_params_,
            "Train RMSE (mean)": np.mean(train_rmse),
            "Train RMSE (std)":  np.std(train_rmse),
            "Val RMSE (mean)":   np.mean(val_rmse),
            "Val RMSE (std)":    np.std(val_rmse),
            "Best Estimator": best_model
        })

    return pd.DataFrame(summary_rows)



#######################################################################################################
# REFIT BEST ESTIMATORS AND COMPUTE TEST RMSE
# ----------------------------------------------------------------------------------------------------
#    Uses the best estimators obtained from evaluate_models_with_grid_search.
#    Each model is retrained on the full training set and evaluated on the test set.
#    No additional cross-validation is performed — RMSE (mean±std) is read from summary_df.
#
#    Parameters:
#    ---------------------------
#    df_train : pd.DataFrame
#        Training dataset (raw, unprocessed).
#
#    df_test : pd.DataFrame
#        Test dataset (raw, unprocessed).
#
#    summary_df : pd.DataFrame
#        Output from evaluate_models_with_grid_search (includes best models and parameters).
#
#    features : list of str
#        List of input features to use for training/testing.
#
#    Returns:
#    ---------------------------
#    final_summary : list of dict
#        Contains:
#        - Cross-validated RMSE (mean±std) read from summary_df
#        - Test RMSE freshly computed on the test set
#
#    trained_models : dict
#        Trained pipeline per model name.
#
#    (X_train, y_train, X_test, y_test) : tuple
#        Transformed input/output arrays for further evaluation or plotting.
#######################################################################################################

def refit_models(df_train, df_test, summary_df, features):

    # ------------------------------------------------------------------------------------------------
    # Prepare train and test data
    # ------------------------------------------------------------------------------------------------
    X_train, y_train, X_test, y_test, preprocessor = prepare_data(
        df_train,
        features=features,
        df_test=df_test,
    )

    # ------------------------------------------------------------------------------------------------
    # Define supported regression models
    # ------------------------------------------------------------------------------------------------
    model_map = {
        "Linear Regression": LinearRegression,
        "KNN": KNeighborsRegressor,
        "Decision Tree": DecisionTreeRegressor,
        "SVR": SVR,
        "MLP": MLPRegressor,
        "Group-Wise Linear": GroupWiseLinearRegressor
    }

    final_summary  = []
    trained_models = {}

    # ------------------------------------------------------------------------------------------------
    # Loop over models and reuse training RMSE from summary_df
    # ------------------------------------------------------------------------------------------------
    for _, row in summary_df.iterrows():
        model_name = row["Model"]
        if model_name not in model_map:
            continue

        # Parse best hyperparameters
        best_params = row["Best Params"]
        if isinstance(best_params, str):
            try:
                best_params = ast.literal_eval(best_params)
            except Exception:
                best_params = {}

        model_class = model_map[model_name]

        # Build model pipeline
        if model_name == "Group-Wise Linear":
            model = model_class(**best_params)
        else:
            clean_params = {k.replace("regressor__", ""): v for k, v in best_params.items()}
            base_model   = model_class(**clean_params)
            model = Pipeline([
                ("preprocessor", preprocessor),
                ("regressor", base_model)
            ])

        # Refit on full train set and evaluate on test set
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        test_rmse   = np.sqrt(mean_squared_error(y_test, y_pred_test))

        # Retrieve stored train metrics
        final_summary.append({
            "Model": model_name,
            "Best Params": row["Best Params"],
            "Train RMSE (CV mean)": row["Train RMSE (mean)"],
            "Train RMSE (CV std)":  row["Train RMSE (std)"],
            "Val RMSE (CV mean)":   row["Val RMSE (mean)"],
            "Val RMSE (CV std)":    row["Val RMSE (std)"],
            "Test RMSE": test_rmse
        })

        trained_models[model_name] = model

    return final_summary, trained_models, (X_train, y_train, X_test, y_test)
