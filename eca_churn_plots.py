from __future__ import annotations

import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = SCRIPT_DIR
INPUT_FILE = BASE_DIR / "ECA_churn_cleaned.csv"
OUTPUT_DIR = BASE_DIR / "figures"


def save_churn_distribution(df: pd.DataFrame) -> Path:
    # Figure 1: Plot the overall churn distribution from the processed dataset.
    counts = df["churn_flag"].value_counts().reindex(["active", "churned"])
    percentages = (counts / counts.sum() * 100).round(2)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color=["#4C956C", "#D1495B"], edgecolor="black")

    ax.set_title("Figure 1. Customer Churn Distribution", fontsize=14, pad=12)
    ax.set_xlabel("Churn Status")
    ax.set_ylabel("Number of Customers")
    ax.set_ylim(0, counts.max() * 1.15)

    for bar, count, pct in zip(bars, counts.values, percentages.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            f"{count}\n({pct}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    output_path = OUTPUT_DIR / "figure_1_churn_distribution.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_average_churn_rate_by_age_group(df: pd.DataFrame) -> Path:
    # Figure 2: Explore churn flag against customer age using average churn rate by age group.
    plot_df = df.copy()
    age_bins = [18, 30, 40, 50, 60, 70, 80]
    age_labels = ["18-29", "30-39", "40-49", "50-59", "60-69", "70-79"]

    plot_df["age_group"] = pd.cut(
        plot_df["customer_age"],
        bins=age_bins,
        labels=age_labels,
        right=False,
        include_lowest=True,
    )

    churn_rate = (
        plot_df.groupby("age_group", observed=False)["churn_flag"]
        .apply(lambda values: (values == "churned").mean() * 100)
        .reindex(age_labels)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        churn_rate.index,
        churn_rate.values,
        color="#2A6F97",
        marker="o",
        linewidth=2.5,
        markersize=8,
    )

    for age_group, rate in churn_rate.items():
        ax.text(age_group, rate + 0.6, f"{rate:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_title("Figure 2. Average Churn Rate by Age Group", fontsize=14, pad=12)
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Average Churn Rate (%)")
    ax.set_ylim(60, 70)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    output_path = OUTPUT_DIR / "figure_2_average_churn_rate_by_age_group.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_churn_rate_heatmap(df: pd.DataFrame) -> Path:
    # Figure 3: Plot churn rate by category and marketing channel as a heatmap.
    heatmap_data = (
        pd.crosstab(
            df["category"],
            df["marketing_channel"],
            values=(df["churn_flag"] == "churned").astype(int),
            aggfunc="mean",
        )
        .mul(100)
        .round(1)
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    heatmap = ax.imshow(heatmap_data.values, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_xticklabels(heatmap_data.columns)
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index)

    for row_index, category in enumerate(heatmap_data.index):
        for col_index, channel in enumerate(heatmap_data.columns):
            ax.text(
                col_index,
                row_index,
                f"{heatmap_data.loc[category, channel]:.1f}%",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )

    ax.set_title("Figure 3. Churn Rate by Category and Marketing Channel", fontsize=14, pad=12)
    ax.set_xlabel("Marketing Channel")
    ax.set_ylabel("Category")
    plt.colorbar(heatmap, ax=ax, label="Churn Rate (%)")

    fig.tight_layout()
    output_path = OUTPUT_DIR / "figure_3_churn_rate_heatmap.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    (BASE_DIR / ".matplotlib").mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(INPUT_FILE)

    output_files = [
        save_churn_distribution(df),
        save_average_churn_rate_by_age_group(df),
        save_churn_rate_heatmap(df),
    ]

    print("Generated three figures from the processed customer churn dataset:")
    for path in output_files:
        print(f"- {path.name}")


if __name__ == "__main__":
    main()
