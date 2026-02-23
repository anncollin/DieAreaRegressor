import numpy as np
import pandas as pd
import re
import os
from sklearn.model_selection import KFold

from Utils.crossValidation import refit_models


def display_final_table(df_train, df_test, all_features, folder="CVRuns", keyword="features"):
    ###################################################################################################
    # PART 1 - COMPUTE SUMMARY VALUES
    ###################################################################################################

    summary_data = {}

    for filename in os.listdir(folder):
        if keyword not in filename or not filename.endswith(".csv"):
            continue

        filepath = os.path.join(folder, filename)

        match = re.search(r"features-\[(.*?)\]\.csv", filename)
        if not match:
            continue

        features = [f.strip() for f in match.group(1).split(",") if f.strip()]

        # Skip configuration with only P_type
        if len(features) == 1 and features[0] == "P_type":
            continue

        # Load CV results and refit
        df_cv = pd.read_csv(filepath)
        summary_df, _, (X_train, y_train, X_test, y_test) = refit_models(
            df_train, df_test, df_cv, features=features
        )

        n_train, n_test = len(X_train), len(X_test)
        feature_flags = tuple("✓" if f in features else "" for f in all_features)

        summary_data[feature_flags] = {
            "#Train": n_train,
            "#Test": n_test
        }

        # Store formatted RMSE strings
        for row in summary_df:
            model = row["Model"]
            summary_data[feature_flags][f"RMSE train ({model})"] = (
                f"{row['Train RMSE (CV mean)']:.2g} ± {row['Train RMSE (CV std)']:.2g}"
            )
            summary_data[feature_flags][f"RMSE val ({model})"] = (
                f"{row['Val RMSE (CV mean)']:.2g} ± {row['Val RMSE (CV std)']:.2g}"
            )
            summary_data[feature_flags][f"RMSE test ({model})"] = f"{row['Test RMSE']:.2g}"




    # Build DataFrame
    rows = []
    for feature_flags, scores in summary_data.items():
        row = {"#Train": scores["#Train"], "#Test": scores["#Test"]}
        row.update({f: v for f, v in zip(all_features, feature_flags)})
        row.update(scores)
        rows.append(row)

    df_result = pd.DataFrame(rows)

    ###################################################################################################
    # PART 2 - ORGANIZE AND SORT TABLE
    ###################################################################################################
    models = ["Linear Regression", "Group-Wise Linear", "KNN", "Decision Tree", "SVR", "MLP"]

    metric_cols = []
    for m in models:
        metric_cols += [
            f"RMSE train ({m})",
            f"RMSE val ({m})",
            f"RMSE test ({m})"
        ]

    df_result = df_result[
        ["#Train", "#Test"] + all_features + [c for c in metric_cols if c in df_result.columns]
    ]

    def feature_sort_key(row):
        count = sum(row[f] == "✓" for f in all_features)
        first = next((i for i, f in enumerate(all_features) if row[f] == "✓"), len(all_features))
        return (count, first)

    df_result["__sort_key__"] = df_result.apply(feature_sort_key, axis=1)
    df_result = (
        df_result.sort_values("__sort_key__")
        .drop(columns="__sort_key__")
        .reset_index(drop=True)
    )

    ###################################################################################################
    # PART 3 - NUMERIC EXTRACTION FOR COLOR MAPS
    ###################################################################################################
    numeric_df = df_result.copy()

    def extract_mean(val):
        if isinstance(val, str) and "±" in val:
            return float(val.split("±")[0].strip())
        try:
            return float(val)
        except Exception:
            return np.nan

    for c in numeric_df.columns:
        if "RMSE" in c:
            numeric_df[c] = numeric_df[c].apply(extract_mean)

    train_cols = [c for c in numeric_df.columns if "RMSE train" in c]
    val_cols   = [c for c in numeric_df.columns if "RMSE val" in c]
    test_cols  = [c for c in numeric_df.columns if "RMSE test" in c]

    ###################################################################################################
    # PART 4 - ADD MEAN COLUMNS AND FINAL ROW
    ###################################################################################################
    formatted_df = df_result.copy()

    formatted_df["Mean train"] = numeric_df[train_cols].mean(axis=1).apply(lambda v: f"{v:.2g}")
    formatted_df["Mean val"]   = numeric_df[val_cols].mean(axis=1).apply(lambda v: f"{v:.2g}")
    formatted_df["Mean test"]  = numeric_df[test_cols].mean(axis=1).apply(lambda v: f"{v:.2g}")

    mean_row = {col: "" for col in formatted_df.columns}
    mean_row["#Train"] = "Mean ↓"

    for c in train_cols + val_cols + test_cols:
        mean_row[c] = f"{numeric_df[c].mean():.2g}"

    mean_row["Mean train"] = f"{numeric_df[train_cols].mean().mean():.2g}"
    mean_row["Mean val"]   = f"{numeric_df[val_cols].mean().mean():.2g}"
    mean_row["Mean test"]  = f"{numeric_df[test_cols].mean().mean():.2g}"

    formatted_df = pd.concat([formatted_df, pd.DataFrame([mean_row])], ignore_index=True)

    formatted_df.to_csv(os.path.join(folder, "final_table.csv"), index=False)

    ###################################################################################################
    # PART 5 - STYLING
    ###################################################################################################
    styled = formatted_df.style
    styled = styled.map(lambda x: "font-weight: bold;" if x == "✓" else "", subset=all_features)

    for col in train_cols:
        styled = styled.background_gradient(cmap="Blues_r", subset=[col], gmap=numeric_df[col])
    for col in val_cols:
        styled = styled.background_gradient(cmap="Oranges_r", subset=[col], gmap=numeric_df[col])

    for col in test_cols:
        styled = styled.background_gradient(cmap="Reds_r", subset=[col], gmap=numeric_df[col])


    styled = styled.set_table_styles([
        {"selector": "th", "props": [("text-align", "center")]},
        {"selector": "td", "props": [("text-align", "center")]}
    ])

    return styled
