from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
import pandas as pd


#######################################################################################################
# GROUP-WISE LINEAR REGRESSOR
# ----------------------------------------------------------------------------------------------------
#    One independent linear model per P_type.
#    If only P_type is present in features: predict the mean Die_area for each P_type.
#    If P_type + numeric features: fit LinearRegression on numeric features for each P_type.
#
#    Notes:
#    ---------------------------
#    - Removes rows with NaN in numeric features before fitting.
#    - Assumes all P_type values seen at prediction were present during training.
#######################################################################################################

class GroupWiseLinearRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.models_ = {}
        self.num_cols_ = []
        self.global_mean_ = None
        self.global_model_ = None

    def fit(self, X, y):
        df = X.copy()
        df["target"] = y.values if hasattr(y, "values") else y

        # Identify numeric columns
        self.num_cols_ = [c for c in X.columns if c != "P_type"]

        # Compute global mean for fallback
        self.global_mean_ = float(df["target"].mean())

        # Compute global linear model as an additional fallback
        if len(self.num_cols_) > 0:
            self.global_model_ = LinearRegression(fit_intercept=self.fit_intercept)
            self.global_model_.fit(df[self.num_cols_], df["target"])

        # Train one model per group
        self.models_ = {}
        for ptype, g in df.groupby("P_type"):
            if len(self.num_cols_) == 0:
                # Only categorical: use mean
                self.models_[ptype] = float(g["target"].mean())
            else:
                # Fit per-type linear regression if enough samples
                if len(g) >= 2:
                    reg = LinearRegression(fit_intercept=self.fit_intercept)
                    reg.fit(g[self.num_cols_], g["target"])
                    self.models_[ptype] = reg
                else:
                    # Too few samples: fallback to global model
                    self.models_[ptype] = self.global_model_

        return self

    def predict(self, X):
        df = X.copy()
        preds = pd.Series(index=df.index, dtype=float)

        # Case 1: Only categorical → map means
        if len(self.num_cols_) == 0:
            preds.loc[df.index] = df["P_type"].map(self.models_)
        else:
            # Case 2: Numeric + categorical
            for ptype, model in self.models_.items():
                idx = df.index[df["P_type"] == ptype]
                if len(idx) == 0:
                    continue
                X_sub = df.loc[idx, self.num_cols_]
                preds.loc[idx] = model.predict(X_sub)

            # Handle unseen P_type → use global model or global mean
            unseen_mask = preds.isna()
            if unseen_mask.any():
                if self.global_model_ is not None:
                    preds.loc[unseen_mask] = self.global_model_.predict(df.loc[unseen_mask, self.num_cols_])
                else:
                    preds.loc[unseen_mask] = self.global_mean_

        # Final safety: replace any remaining NaNs by global mean
        preds = preds.fillna(self.global_mean_)
        return preds.values
