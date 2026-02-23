import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


#######################################################################################################
# LOAD AND PREPROCESS TRAINING + TEST DATA FROM EXCEL
# ----------------------------------------------------------------------------------------------------
#    Parameters:
#    ---------------------------
#    excel_path : str
#        Path to Excel file containing sheets "data_combined" (train) and "data_demonstrator" (test).
#
#    Returns:
#    ---------------------------
#    df_train, df_test : pd.DataFrame
#        Cleaned training and test sets.
#######################################################################################################

def load_data(excel_path="dataset.xlsx"):
    
    selected_columns = {
        "attributed package type (aggregated)": "P_type",
        "pin count": "Pin_count",
        "package area [mm^2]": "P_area",
        "package mass [g]": "M",
        "die area [mm^2]": "Die_area"
    }

    def prepare_dataset(df, is_test=False):
        df = df.copy()

        if not is_test:
            # Remove rows with 'x' in ignore columns
            df = df[
                (df["to ignore (not an IC)"] != "x") &
                (df["to ignore (p-type)"] != "x")
            ].copy()

        # Select and rename columns (without 'isMultiple')
        columns_to_use = list(selected_columns.keys())
        df_selected = df[columns_to_use].rename(columns=selected_columns)

        # Standardize naming convention for package types 
        if "P_type" in df_selected.columns:
            df_selected["P_type"] = df_selected["P_type"].replace("WLCSP", "WLP")
            df_selected["P_type"] = df_selected["P_type"].replace("DIP", "SOP/SOT")

        return df_selected

    # Load Excel sheets
    df_combined     = pd.read_excel(excel_path, sheet_name="data_combined")
    df_demonstrator = pd.read_excel(excel_path, sheet_name="data_demonstrator")

    # Prepare datasets
    df_train = prepare_dataset(df_combined, is_test=False)
    df_test  = prepare_dataset(df_demonstrator, is_test=True)

    df_train = df_train.dropna(subset=["Die_area"])
    df_test  = df_test.dropna(subset=["Die_area"])  

    return df_train, df_test


#######################################################################################################
# PREPARE TRAINING (AND OPTIONAL TEST) DATA FOR MODELING
# ----------------------------------------------------------------------------------------------------
#    Always applies log10 transform to numeric features and target (Die_area).
#    Optionally applies StandardScaler to numeric features.
#
#    Parameters:
#    ---------------------------
#    df_train : pd.DataFrame
#        Training dataframe.
#
#    features : list of str
#        List of feature names to use.
#
#    df_test : pd.DataFrame or None
#        Optional test set (same structure as training set).

#    Returns:
#    ---------------------------
#    If df_test is None:
#        X_train, y_train, preprocessor
#    Else:
#        X_train, y_train, X_test, y_test, preprocessor
#######################################################################################################

def prepare_data(df_train, features, df_test=None):
    df_train = df_train.copy()
    if df_test is not None:
        df_test = df_test.copy()

    # Drop rows with missing values 
    df_train = df_train.dropna(subset=features + ["Die_area"])
    if df_test is not None:
        df_test = df_test.dropna(subset=features + ["Die_area"])

    # Identify numeric and categorical features 
    numeric_features     = [f for f in features if f != "P_type"]
    categorical_features = [f for f in features if f == "P_type"]

    # Apply log10 transform to all numeric features and target 
    for f in numeric_features:
        df_train[f] = np.log10(df_train[f])
        if df_test is not None:
            df_test[f] = np.log10(df_test[f])

    df_train["Die_area"] = np.log10(df_train["Die_area"])
    if df_test is not None:
        df_test["Die_area"] = np.log10(df_test["Die_area"])

    # Split into feature and target matrices 
    X_train = df_train[features].copy()
    y_train = df_train["Die_area"]
    if df_test is not None:
        X_test = df_test[features].copy()
        y_test = df_test["Die_area"]

    # Define preprocessing pipeline: only encode categorical features
    transformers = []
    if len(categorical_features) > 0:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features))

    preprocessor = ColumnTransformer(transformers, remainder="passthrough")

    # Return depending on whether a test set is provided 
    if df_test is not None:
        return X_train, y_train, X_test, y_test, preprocessor
    else:
        return X_train, y_train, preprocessor

