import numpy as np
import pandas as pd
import re
import os
from sklearn.model_selection import KFold

from Utils.crossValidation import refit_models


#######################################################################################################
# DISPLAY FINAL PERFORMANCE TABLE (TRAIN + TEST)
# ----------------------------------------------------------------------------------------------------
#    Aggregates results from CSV files in `folder`, showing RMSE (train/test) for each feature combo.
#
#    Parameters:
#    ---------------------------
#    df_train : pd.DataFrame
#        Training dataset.
#
#    df_test : pd.DataFrame
#        Test dataset.
#
#    all_features : list of str
#        All possible feature names (columns).
#
#    folder : str, default="CVRuns"
#        Folder containing saved CSV results.
#
#    keyword : str, default="final"
#        Keyword that must appear in filenames to be considered.
#
#    Returns:
#    ---------------------------
#    styled : pd.io.formats.style.Styler
#        Styled DataFrame (ready for display).
#######################################################################################################

import os
import re
import pandas as pd
import numpy as np


#######################################################################################################
# DISPLAY FINAL PERFORMANCE TABLE (TRAIN + TEST)
#######################################################################################################
def display_final_table(df_train, df_test, all_features, folder="CVRuns", keyword="features"):
    ###################################################################################################
    # PART 1 - COMPUTE SUMMARY VALUES
    ###################################################################################################

    summary_data = {}

    for filename in os.listdir(folder):
        if keyword not in filename or not filename.endswith(".csv"):
            continue

        filepath = os.path.join(folder, filename)

        # Match new filename pattern: features-[M,P_area,P_type].csv
        match = re.search(r"features-\[(.*?)\]\.csv", filename)
        if not match:
            continue

        features_str = match.group(1)
        features = [f.strip() for f in features_str.split(",") if f.strip()]

        # Load CSV and refit models
        df_final = pd.read_csv(filepath)
        summary_df, _, (X_train, y_train, X_test, y_test) = refit_models(
            df_train, df_test, df_final,
            features=features
        )

        n_train, n_test = len(X_train), len(X_test)
        feature_flags = tuple("✓" if f in features else "" for f in all_features)
        summary_data[feature_flags] = {"#Train": n_train, "#Test": n_test}

        # Store RMSE metrics
        for row in summary_df:
            model = row["Model"]
            mean_val = row.get("Train RMSE (CV mean)")
            std_val = row.get("Train RMSE (CV std)")
            test_val = row.get("Test RMSE")
            summary_data[feature_flags][f"RMSE train ({model})"] = (mean_val, std_val)
            summary_data[feature_flags][f"RMSE test ({model})"] = test_val

    # Convert summary into DataFrame
    rows = []
    for feature_flags, model_scores in summary_data.items():
        row = {"#Train": model_scores["#Train"], "#Test": model_scores["#Test"]}
        row.update({f: v for f, v in zip(all_features, feature_flags)})
        row.update(model_scores)
        rows.append(row)

    df_result = pd.DataFrame(rows)

    ###################################################################################################
    # PART 2 - ORGANIZE AND SORT TABLE
    ###################################################################################################
    models = ["Linear Regression", "Group-Wise Linear", "KNN", "Decision Tree", "SVR", "MLP"]
    metric_cols = []
    for m in models:
        metric_cols += [f"RMSE train ({m})", f"RMSE test ({m})"]

    df_result = df_result[["#Train", "#Test"] + all_features + [c for c in metric_cols if c in df_result.columns]]

    # Sort rows by number and order of features
    def feature_sort_key(row):
        count = sum(row[f] == "✓" for f in all_features)
        first = next((i for i, f in enumerate(all_features) if row[f] == "✓"), len(all_features))
        return (count, first)

    df_result["__sort_key__"] = df_result.apply(feature_sort_key, axis=1)
    df_result = df_result.sort_values("__sort_key__").drop(columns="__sort_key__").reset_index(drop=True)

    ###################################################################################################
    # PART 3 - FORMAT AND STYLE TABLE
    ###################################################################################################
    formatted_df = df_result.copy()

    # Format numeric cells (train mean ± std, test single value)
    for c in formatted_df.columns:
        if "train" in c:
            formatted_df[c] = formatted_df[c].apply(
                lambda v: f"{v[0]:.2g} ± {v[1]:.2g}" if isinstance(v, tuple) and all(not pd.isna(x) for x in v) else ""
            )
        elif "test" in c:
            formatted_df[c] = formatted_df[c].apply(
                lambda v: f"{v:.2g}" if isinstance(v, (float, int)) and not pd.isna(v) else ""
            )

    # Extract numeric values for color gradients
    numeric_df = df_result.copy()
    for c in numeric_df.columns:
        if "train" in c:
            numeric_df[c] = numeric_df[c].apply(lambda v: v[0] if isinstance(v, tuple) else np.nan)
        elif "test" in c:
            numeric_df[c] = numeric_df[c].apply(lambda v: v if isinstance(v, (float, int)) else np.nan)

    train_cols = [c for c in numeric_df.columns if "train" in c]
    test_cols  = [c for c in numeric_df.columns if "test" in c]

    # Add per-row mean values
    formatted_df["Mean train"] = numeric_df[train_cols].mean(axis=1, skipna=True).apply(lambda v: f"{v:.2g}" if not pd.isna(v) else "")
    formatted_df["Mean test"]  = numeric_df[test_cols].mean(axis=1, skipna=True).apply(lambda v: f"{v:.2g}" if not pd.isna(v) else "")

    # Add final mean row (white cells + black text)
    mean_row = {col: "" for col in formatted_df.columns}
    mean_row["#Train"] = "Mean ↓"
    for c in train_cols:
        mean_row[c] = f"{numeric_df[c].mean(skipna=True):.2g}"
    for c in test_cols:
        mean_row[c] = f"{numeric_df[c].mean(skipna=True):.2g}"
    mean_row["Mean train"] = f"{numeric_df[train_cols].mean(skipna=True).mean():.2g}"
    mean_row["Mean test"]  = f"{numeric_df[test_cols].mean(skipna=True).mean():.2g}"
    formatted_df = pd.concat([formatted_df, pd.DataFrame([mean_row])], ignore_index=True)

    formatted_df.to_csv(os.path.join(folder, "final_table.csv"), index=False)

    ###################################################################################################
    # PART 4 - STYLING (BLUE/RED SIDE-BY-SIDE + WHITE LAST ROW)
    ###################################################################################################
    def color_nan(val):
        return "background-color: lightgray;" if val in ["", None] or pd.isna(val) else ""

    styled = formatted_df.style
    styled = styled.map(lambda x: "font-weight: bold; color: black;" if x == "✓" else "", subset=all_features)

    # Apply gradients for all rows except the last (mean row)
    cmap_train, cmap_test = "Blues_r", "Reds_r"
    last_index = len(formatted_df) - 1

    for col in train_cols:
        styled = styled.background_gradient(subset=[col], cmap=cmap_train, gmap=numeric_df[col])
    for col in test_cols:
        styled = styled.background_gradient(subset=[col], cmap=cmap_test, gmap=numeric_df[col])

    # Gray NaNs
    styled = styled.map(color_nan, subset=train_cols + test_cols)

    # White background for last row (mean row)
    styled = styled.map_index(
        lambda idx: "background-color: white; color: black;" if idx == last_index else "",
        axis=0
    )

    styled = styled.set_table_styles([
        {"selector": "th", "props": [("white-space", "pre-line"), ("text-align", "center")]},
        {"selector": "td", "props": [("text-align", "center")]}
    ])

    return styled

