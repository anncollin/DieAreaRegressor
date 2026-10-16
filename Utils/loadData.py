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
#        Path to Excel files containing sheets with train and test data.
#        The dataset is divided into two parts: the UCLouvain part and the Fraunhofer part.
#        The UCLouvain part contains data for the training set and the test set. The Fraunhofer part was used only in the train set. 
#
#    Returns:
#    ---------------------------
#    df_train, df_test : pd.DataFrame
#        Cleaned training and test sets.
#######################################################################################################

def load_data(excel_path_1="data/OSSDA_021726_UCLOUVAIN.xlsx", excel_path_2="data/OSSDA_021726_FRAUNHOFER_IZM.xlsx"):
    
    selected_columns = {
        "attributed package type (aggregated)": "P_type",
        "pin count": "Pin_count",
        "package area [mm^2]": "P_area",
        "package mass [g]": "M",
        "die area [mm^2]": "Die_area"
    }

    def prepare_dataset(df, is_test=False):
        df = df.copy()

        # Select and rename columns (without 'isMultiple')
        columns_to_use = list(selected_columns.keys())
        df_selected = df[columns_to_use].rename(columns=selected_columns)

        # Standardize naming convention for package types 
        if "P_type" in df_selected.columns:
            df_selected["P_type"] = df_selected["P_type"].replace("WLCSP", "WLP")
            df_selected["P_type"] = df_selected["P_type"].replace("DIP", "SOP/SOT")

        return df_selected

    # Load Excel sheets
    df_train_1    = pd.read_excel(excel_path_1, sheet_name="OSSDA_train")
    df_test_1 = pd.read_excel(excel_path_1, sheet_name="OSSDA_test")
    df_train_2 = pd.read_excel(excel_path_2, sheet_name="OSSDA_train")
    df_combined_train = pd.concat([df_train_1, df_train_2], ignore_index=True, sort=False)
    #print(df_combined_train)

    # Prepare datasets
    df_train = prepare_dataset(df_combined_train, is_test=False)
    df_test  = prepare_dataset(df_test_1, is_test=True)

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

