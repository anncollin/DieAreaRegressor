import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from Utils.loadData import prepare_data

#######################################################################################################
# PRINT_GROUPWISE_LINEAR_EQUATIONS (log₁₀ scale)
# ----------------------------------------------------------------------------------------------------
#    Parameters:
#    ---------------------------
#    df_train : pd.DataFrame
#        Input DataFrame containing numeric features, 'P_type', and 'Die_area'.
#
#    all_features : list of str
#        List of features to include (must contain 'P_type' as grouping feature).
#
#    Returns:
#    ---------------------------
#    group_models : dict
#        Dictionary mapping each P_type group to its fitted LinearRegression model.
#######################################################################################################

def print_groupwise_linear_equations(df_train, all_features):

    target_col    = "Die_area"
    group_feature = "P_type"

    if group_feature not in df_train.columns:
        raise ValueError("The dataframe must contain a 'P_type' column.")

    numeric_features = [f for f in all_features if f != group_feature]
    if len(numeric_features) == 0:
        raise ValueError("At least one numeric feature is required for regression.")

    group_models = {}

    print("\nParametric equations of Group-Wise Linear Regressor (log₁₀ scale):\n")

    for group_value, group_data in df_train.groupby(group_feature):

        # Apply log₁₀ transform to numeric features and target
        X = np.log10(group_data[numeric_features].astype(float))
        y = np.log10(group_data[target_col].astype(float))

        model = LinearRegression().fit(X, y)
        group_models[group_value] = model

        intercept = model.intercept_
        coefs     = model.coef_

        eq_terms = " + ".join([f"{coef:.2g}·log₁₀({name})" for coef, name in zip(coefs, numeric_features)])
        eq_str = f"log₁₀({target_col}) = {intercept:.2g} + {eq_terms}"

        print(f"Group '{group_value}':")
        print(f"    {eq_str}\n")

    return group_models


#######################################################################################################
# PRINT_LINEAR_EQUATION (log₁₀ scale)
# ----------------------------------------------------------------------------------------------------
#    Parameters:
#    ---------------------------
#    df_train : pd.DataFrame
#        Input DataFrame containing numeric features and 'Die_area'.
#
#    all_features : list of str
#        List of features to include (may contain 'P_type' — ignored if present).
#
#    Returns:
#    ---------------------------
#    model : sklearn.linear_model.LinearRegression
#        Fitted Linear Regression model on log₁₀-transformed data.
#######################################################################################################

def print_linear_equation(df_train, all_features):

    target_col = "Die_area"

    # Prepare data (handles NaNs, log10 transform, encoding)
    X_train, y_train, preprocessor = prepare_data(df_train, all_features)

    # Transform and fit model
    X_processed = preprocessor.fit_transform(X_train)
    model       = LinearRegression().fit(X_processed, y_train)

    # Retrieve feature names after preprocessing
    feature_names = preprocessor.get_feature_names_out(all_features)

    # Identify numeric features
    numeric_features = [f for f in all_features if f != "P_type"]

    # Clean names for readability
    clean_names = []
    for name in feature_names:
        if name.startswith("cat__P_type_"):
            clean_names.append(name.replace("cat__P_type_", ""))  # remove prefix for clarity
        elif name.startswith("remainder__"):
            base_name = name.replace("remainder__", "")
            if base_name in numeric_features:
                clean_names.append(f"log₁₀({base_name})")          # explicit log10()
            else:
                clean_names.append(base_name)
        else:
            clean_names.append(name)

    # Build regression equation
    intercept = model.intercept_
    coefs = model.coef_

    eq_terms = " + ".join([f"{coef:.2g}·{name}" for coef, name in zip(coefs, clean_names)])
    eq_str = f"log₁₀({target_col}) = {intercept:.2g} + {eq_terms}"

    print(eq_str)

    # Informational note if categorical feature is used
    if "P_type" in all_features:
        print("\nNote: The categorical variable 'P_type' is one-hot encoded.")
        print("      Each coefficient corresponds to the contribution of a specific package type\n")

    return model