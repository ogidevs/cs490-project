import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("ggplot")
sns.set_theme(style="whitegrid")


def plot_distributions(df):
    """Plots histogram distribution curves for pricing and physical area metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(df["Total_Price_EUR"], bins=40, kde=True, color="blue", ax=axes[0])
    axes[0].set_title("Distribution: Total Price (€)")

    sns.histplot(df["Area"], bins=40, kde=True, color="green", ax=axes[1])
    axes[1].set_title("Distribution: Area (m²)")

    if "Price_per_Unit_EUR" in df.columns:
        sns.histplot(
            df["Price_per_Unit_EUR"], bins=40, kde=True, color="purple", ax=axes[2]
        )
        axes[2].set_title("Distribution: Price / m²")

    plt.tight_layout()
    return fig


def plot_relationships(df):
    """Plots the relationship between scale (area) and cost, enforcing a regression curve."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    if "City" in df.columns and df["City"].nunique() > 1:
        sns.scatterplot(
            x="Area",
            y="Total_Price_EUR",
            hue="City",
            data=df,
            alpha=0.7,
            ax=axes[0],
            palette="tab10",
        )
    else:
        sns.regplot(
            x="Area",
            y="Total_Price_EUR",
            data=df,
            scatter_kws={"alpha": 0.6},
            line_kws={"color": "red"},
            ax=axes[0],
        )

    axes[0].set_title("Area vs. Total Price")
    axes[0].set_xlabel("Area (m²)")
    axes[0].set_ylabel("Total Price (€)")

    if "City" in df.columns:
        sns.boxplot(
            x="City",
            y="Price_per_Unit_EUR",
            hue="City",
            data=df,
            ax=axes[1],
            palette="Set2",
            legend=False,
        )
        axes[1].set_title("Price per m² Variance by City")
        axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    return fig


def plot_advanced_features(df):
    """Groups categorical elements like Rooms and Municipalities to display their price impacts."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    if "Rooms_Numeric" in df.columns:
        df_viz = df.copy()
        df_viz["Rooms_Cat"] = df_viz["Rooms_Numeric"].apply(
            lambda x: "5+" if pd.notnull(x) and x >= 5 else str(x)
        )
        sns.violinplot(
            x="Rooms_Cat",
            y="Total_Price_EUR",
            hue="Rooms_Cat",
            data=df_viz,
            ax=axes[0],
            order=["0.5", "1.0", "1.5", "2.0", "2.5", "3.0", "3.5", "4.0", "4.5", "5+"],
            palette="muted",
            legend=False,
            dodge=False,
        )
        axes[0].set_title("Price Variance by Room Count")

    if "Municipality" in df.columns:
        top_muni = df["Municipality"].value_counts().nlargest(10).index
        df_top = df[df["Municipality"].isin(top_muni)]
        sns.barplot(
            x="Municipality",
            y="Total_Price_EUR",
            hue="Municipality",
            data=df_top,
            estimator=np.median,
            ax=axes[1],
            errorbar=None,
            palette="viridis",
            legend=False,
            dodge=False,
        )
        axes[1].set_title("Median Price in Top 10 Active Municipalities")
        axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    return fig


def plot_floor_impact(df):
    """Analyzes how the property's floor level impacts its value."""
    fig, ax = plt.subplots(figsize=(10, 6))
    if "Current_Floor_Num" in df.columns:
        # Group floors to prevent clutter
        df_viz = df.copy()
        df_viz["Floor_Group"] = pd.cut(
            df_viz["Current_Floor_Num"],
            bins=[-2, 0, 3, 8, 100],
            labels=["Basement/Ground", "Low (1-3)", "Mid (4-8)", "High (9+)"],
        )
        sns.barplot(
            x="Floor_Group",
            y="Price_per_Unit_EUR",
            data=df_viz,
            estimator=np.median,
            errorbar=None,
            palette="magma",
            ax=ax,
        )
        ax.set_title("Median Price per m² based on Floor Level")
    return fig


def plot_correlation_matrix(df):
    """Calculates cross-vector tracking matrix indicating feature synergy."""
    num_cols = [
        "Total_Price_EUR",
        "Area",
        "Price_per_Unit_EUR",
        "Rooms_Numeric",
        "Current_Floor_Num",
        "Total_Floors_Num",
        "Photo_Count",
    ]
    available_cols = [c for c in num_cols if c in df.columns]

    corr_matrix = df[available_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, vmin=-1, vmax=1
    )
    ax.set_title("Feature Correlation Profile")
    plt.tight_layout()
    return fig
