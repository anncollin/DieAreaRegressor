import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import pearsonr


#######################################################################################################
# PLOT DISTRIBUTIONS AND CORRELATIONS FOR ALL NUMERIC FEATURES (EXCEPT 'Die_area')
# ----------------------------------------------------------------------------------------------------
#    Parameters:
#    ---------------------------
#    df_train : pd.DataFrame
#        Training dataset (must include numeric features and 'Die_area').
#
#    df_test : pd.DataFrame
#        Test dataset (same structure as df_train).
#
#    Description:
#    ---------------------------
#    For each numeric feature:
#        - Column 1: Histogram (linear x-axis)
#        - Column 2: Scatter vs Die_area (linear x-axis)
#        - Column 3: Histogram (log x-axis)
#        - Column 4: Scatter vs Die_area (log x-axis)
#######################################################################################################

def plot_numeric_distributions_grid(df_train, df_test):

    # Select numeric features except the target
    numeric_cols = df_train.select_dtypes(include='number').columns.drop('Die_area')

    for col in numeric_cols:
        _, axes = plt.subplots(1, 4, figsize=(20, 4))

        # Prepare data
        train_vals = df_train[col].dropna()
        test_vals  = df_test[col].dropna()
        all_vals   = pd.concat([train_vals, test_vals])

        # -------------------------
        # 1. Histogram (linear x)
        # -------------------------
        lin_bins = np.linspace(all_vals.min(), all_vals.max(), 30)
        axes[0].hist(train_vals, bins=lin_bins, color='steelblue', alpha=0.6, label='Train', density=False)
        axes[0].hist(test_vals, bins=lin_bins, color='orange', alpha=0.6, label='Test', density=False)
        axes[0].set_title(f'Hist (linear) — {col}')
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('# of samples')
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.5)

        # -------------------------
        # 2. Scatter (linear x)
        # -------------------------
        axes[1].scatter(train_vals, df_train.loc[train_vals.index, 'Die_area'], color='steelblue', alpha=0.6, label='Train')
        axes[1].scatter(test_vals, df_test.loc[test_vals.index, 'Die_area'], color='orange', alpha=0.6, label='Test')
        axes[1].set_title(f'Scatter (linear-x) — {col}')
        axes[1].set_xlabel(col)
        axes[1].set_ylabel('Die_area')
        axes[1].legend()
        axes[1].grid(True, linestyle='--', alpha=0.5)

        # -------------------------
        # 3. Histogram (log x)
        # -------------------------
        bins = np.logspace(
            np.log10(min(train_vals.min(), test_vals.min())),
            np.log10(max(train_vals.max(), test_vals.max())),
            30
        )

        axes[2].hist(train_vals, bins=bins, color='steelblue', alpha=0.6, label='Train', density=False)
        axes[2].hist(test_vals, bins=bins, color='orange', alpha=0.6, label='Test', density=False)
        axes[2].set_title(f'Hist (log-x) — {col}')
        axes[2].set_xlabel(f'log({col})')
        axes[2].set_ylabel('# of samples')
        axes[2].set_xscale('log')
        axes[2].grid(True, which='major', linestyle='--', linewidth=0.8, alpha=0.6)
        axes[2].grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.4)
        axes[2].legend()


        # -------------------------
        # 4. Scatter (log - log)
        # -------------------------
        axes[3].scatter(train_vals, df_train.loc[train_vals.index, 'Die_area'], color='steelblue', alpha=0.6, label='Train')
        axes[3].scatter(test_vals, df_test.loc[test_vals.index, 'Die_area'], color='orange', alpha=0.6, label='Test')
        axes[3].set_xscale('log')
        axes[3].set_title(f'Scatter (log-x) — {col}')
        axes[3].set_xlabel(f'log({col})')
        axes[3].set_yscale('log')
        axes[3].set_ylabel('log(Die_area)')
        axes[3].grid(True, which='major', linestyle='--', linewidth=0.8, alpha=0.6)
        axes[3].grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.4)
        axes[3].legend()

        plt.tight_layout()
        plt.show()


#######################################################################################################
# BOXPLOT + STRIP PLOT OF DIE_AREA BY PACKAGE TYPE (P_type)
# ----------------------------------------------------------------------------------------------------
#    Parameters:
#    ---------------------------
#    df_train : pd.DataFrame
#        Training dataset with 'P_type' and 'Die_area'.
#
#    df_test : pd.DataFrame
#        Test dataset with same features.
#
#    log : bool
#        If True, plot Die_area on a log scale.
#
#    Description:
#    ---------------------------
#    - Boxplot shows training distribution of Die_area per P_type
#    - Overlaid strip plots display individual points from both train and test sets
#######################################################################################################

def plot_box_strip_by_ptype(df_train, df_test, log=False):

    # Label datasets
    df_train_labeled = df_train.copy()
    df_train_labeled["Dataset"] = "Train"

    df_test_labeled = df_test.copy()
    df_test_labeled["Dataset"] = "Test"

    # Combine for boxplot
    df_combined = pd.concat([df_train_labeled, df_test_labeled], ignore_index=True)

    plt.figure(figsize=(10, 6))

    # Boxplot
    sns.boxplot(
        data=df_combined,
        x='P_type', y='Die_area', hue='Dataset',
        showcaps=True, showfliers=False, width=0.5,
        boxprops={'alpha': 0.6},
        whiskerprops={'linewidth': 1.5}
    )

    # Retrieve categories in correct display order
    ax = plt.gca()
    categories = [t.get_text() for t in ax.get_xticklabels()]
    offset = 0.18

    # Plot per-category aligned points
    for i, cat in enumerate(categories):
        # Train points (blue)
        subset_train = df_train_labeled[df_train_labeled["P_type"] == cat]
        xvals_train = np.random.normal(i - offset, 0.04, size=len(subset_train))
        plt.scatter(xvals_train, subset_train["Die_area"], color='steelblue', alpha=0.5, s=20)

        # Test points (orange)
        subset_test = df_test_labeled[df_test_labeled["P_type"] == cat]
        xvals_test = np.random.normal(i + offset, 0.04, size=len(subset_test))
        plt.scatter(xvals_test, subset_test["Die_area"], color='orange', alpha=0.7, s=20)

    # Labels, legend, and grid
    plt.title('Die_area Distribution by P_type')
    plt.xlabel('P_type')
    plt.ylabel('Die_area (log scale)' if log else 'Die_area')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)

    if log:
        plt.yscale("log")

    # Add counts above x-ticks (same y position for all)
    y_min, y_max = ax.get_ylim()
    y_text = y_min * 0.8 if log else y_min - (y_max - y_min) * 0.08  # consistent y-position

    for i, cat in enumerate(categories):
        n_train = len(df_train_labeled[df_train_labeled["P_type"] == cat])
        n_test  = len(df_test_labeled[df_test_labeled["P_type"] == cat])

        plt.text(i - offset, y_text, str(n_train),
                 color='steelblue', ha='center', va='top', fontsize=12, fontweight='bold')
        plt.text(i + offset, y_text, str(n_test),
                 color='orange', ha='center', va='top', fontsize=12, fontweight='bold')

    # Adjust limits so text fits
    if log:
        ax.set_ylim(y_min * 0.5, y_max)
    else:
        ax.set_ylim(y_min - (y_max - y_min) * 0.15, y_max)

    # Simplify legend
    handles, labels = ax.get_legend_handles_labels()
    ax.grid(True, which='major', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.4)
    plt.legend(handles[:2], labels[:2], title='Dataset')
    plt.tight_layout()
    plt.show()


#######################################################################################################
# DENSITY PLOTS OF DIE_AREA (LINEAR + LOG SCALE)
# ----------------------------------------------------------------------------------------------------
#    Parameters:
#    ---------------------------
#    df_train : pd.DataFrame
#        Training set containing 'Die_area'.
#
#    df_test : pd.DataFrame
#        Test set containing 'Die_area'.
#
#    Description:
#    ---------------------------
#    - Plots side-by-side histograms of Die_area in linear and log scale
#    - Compares train and test distributions
#######################################################################################################

def plot_Die_area_distribution(df_train, df_test):

    # Compute shared min and max across train and test
    min_val = min(df_train['Die_area'].min(), df_test['Die_area'].min())
    max_val = max(df_train['Die_area'].max(), df_test['Die_area'].max())

    # Linear bins
    bins_linear = np.linspace(min_val, max_val, 30)

    # Logarithmic bins
    bins_log = np.logspace(np.log10(min_val), np.log10(max_val), 30)

    # Plot
    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ---------------------------
    # Left: Linear scale (counts)
    # ---------------------------
    sns.histplot(df_train['Die_area'], bins=bins_linear, color='steelblue',
                 label='Train', alpha=0.6, ax=axes[0], edgecolor=None)
    sns.histplot(df_test['Die_area'], bins=bins_linear, color='orange',
                 label='Test', alpha=0.6, ax=axes[0], edgecolor=None)
    axes[0].set_title("Die_area Distribution (Linear Scale)")
    axes[0].set_xlabel("Die_area")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # ------------------------------------------
    # Right: Log scale (counts, true log-x axis)
    # ------------------------------------------
    sns.histplot(df_train['Die_area'], bins=bins_log, color='steelblue',
                 label='Train', alpha=0.6, ax=axes[1])
    sns.histplot(df_test['Die_area'], bins=bins_log, color='orange',
                 label='Test', alpha=0.6, ax=axes[1])
    axes[1].set_xscale('log')
    axes[1].set_title("Die_area Distribution (Log X-Scale)")
    axes[1].set_xlabel("Die_area (log scale)")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    axes[1].grid(True, which='major', linestyle='--', linewidth=0.8, alpha=0.6)
    axes[1].grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.4)
    plt.tight_layout()
    plt.show()


#######################################################################################################
# PLOT_PAIRPLOT_WITH_CORRELATION — Pairplot with Pearson correlations and optional log10 transformation
# ----------------------------------------------------------------------------------------------------
#    Parameters:
#    ---------------------------
#    df_train : pd.DataFrame
#        The cleaned training dataset.
#
#    features : list of str
#        Feature names to include in the pairplot.
#
#    log_log : bool
#        If True, apply log10 transformation to features (data + labels).
#
#    Description:
#    ---------------------------
#    - Creates a Seaborn pairplot colored by P_type.
#    - Adds Pearson correlation coefficients on all non-diagonal scatter plots.
#    - If log_log=True, uses log10-transformed data and renames axes to log(var).
#    - Displays legend outside the plot area on the right.
#######################################################################################################

def plot_pairplot_with_correlation(df_train, features, log_log=False):

    # Copy dataset and drop NaNs
    df = df_train.copy()
    df = df.dropna(subset=features + ["P_type"])

    # Log10 transform features if requested
    if log_log:
        for f in features:
            df[f] = np.log10(df[f])
        features_log = [f"log({f})" for f in features]
        df.rename(columns=dict(zip(features, features_log)), inplace=True)
        features = features_log

    # Define color palette by P_type
    p_type_levels = sorted(df["P_type"].unique())
    palette = sns.color_palette("Spectral", len(p_type_levels))
    palette_dict = dict(zip(p_type_levels, palette))

    # Create Seaborn pairplot
    g = sns.pairplot(
        df,
        vars=features,
        hue="P_type",
        palette=palette_dict,
        corner=False,
        plot_kws={'alpha': 0.5, 'edgecolor': 'none'}
    )

    # Add Pearson correlations (computed on current data — log if log_log=True)
    for i, var_row in enumerate(features):
        for j, var_col in enumerate(features):
            if i != j:
                ax = g.axes[i, j]
                subset = df[[var_col, var_row]].dropna()
                if len(subset) > 1:
                    r, _ = pearsonr(subset[var_col], subset[var_row])
                    ax.annotate(f"r = {r:.2f}",
                                xy=(0.5, 0.9),
                                xycoords='axes fraction',
                                ha='center',
                                fontsize=10,
                                fontweight='bold',
                                color='black')

    # Adjust title and legend
    title = "Pairplot of Features Colored by P_type (log10 scale)" if log_log else "Pairplot of Features Colored by P_type"
    g.fig.suptitle(title, y=1.03, fontsize=16)
    handles = g._legend_data.values()
    labels = g._legend_data.keys()
    g.fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.01, 0.5), title="P_type")
    g._legend.remove()

    plt.tight_layout()
    plt.show()