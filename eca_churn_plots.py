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

CONTINENT_MAP = {
    "Australia": "Oceania",
    "Bahrain": "Asia",
    "Canada": "North America",
    "France": "Europe",
    "Germany": "Europe",
    "Ireland": "Europe",
    "Italy": "Europe",
    "Netherlands": "Europe",
    "Sweden": "Europe",
    "Switzerland": "Europe",
    "United Arab Emirates": "Asia",
    "United Kingdom": "Europe",
}

CONTINENT_COLORS = {
    "Asia": "#D1495B",
    "Europe": "#2A6F97",
    "North America": "#4C956C",
    "Oceania": "#E09F3E",
    "Other": "#7A7A7A",
}


def save_churn_rate_by_subcategory(df: pd.DataFrame) -> Path:
    # Figure 1: Compare average churn rate across product subcategories.
    churn_rate = (
        df.groupby("subcategory")["churn_flag"]
        .apply(lambda values: (values == "churned").mean() * 100)
        .sort_values(ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(
        churn_rate.index,
        churn_rate.values,
        color="#2A6F97",
        edgecolor="black",
    )

    for bar, rate in zip(bars, churn_rate.values):
        label_x = max(min(rate - 0.2, 69.8), 55.8)
        ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"{rate:.1f}%",
            ha="right",
            va="center",
            fontsize=9,
            color="white",
            clip_on=True,
        )

    ax.set_title("Figure 1. Churn Rate by Subcategory", fontsize=14, pad=12)
    ax.set_xlabel("Average Churn Rate (%)")
    ax.set_ylabel("Subcategory")
    ax.set_xlim(55, 70)
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    fig.tight_layout()
    output_path = OUTPUT_DIR / "figure_1_churn_rate_by_subcategory.png"
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


def save_churn_rate_by_delivery_time(df: pd.DataFrame) -> Path:
    # Figure 3: Explore how average churn rate changes across delivery time values.
    churn_rate = (
        df.groupby("delivery_time_days")["churn_flag"]
        .apply(lambda values: (values == "churned").mean() * 100)
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(
        churn_rate.index,
        churn_rate.values,
        color="#D1495B",
        marker="o",
        linewidth=2.5,
        markersize=8,
    )

    for delivery_time, rate in churn_rate.items():
        ax.text(
            delivery_time,
            rate + 0.6,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_title("Figure 3. Churn Rate by Delivery Time", fontsize=14, pad=12)
    ax.set_xlabel("Delivery Time (Days)")
    ax.set_ylabel("Average Churn Rate (%)")
    ax.set_xticks(churn_rate.index.tolist())
    ax.set_ylim(55, 70)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    output_path = OUTPUT_DIR / "figure_3_churn_rate_by_delivery_time.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_churn_rate_heatmap_by_segment_channel(df: pd.DataFrame) -> Path:
    # Figure 4: Heatmap of churn rate by customer segment and marketing channel.
    plot_df = df.loc[df["customer_segment"] != "Missing"].copy()
    heatmap_data = (
        pd.crosstab(
            plot_df["customer_segment"],
            plot_df["marketing_channel"],
            values=(plot_df["churn_flag"] == "churned").astype(int),
            aggfunc="mean",
        )
        .mul(100)
        .reindex(index=["New", "Regular", "VIP"], columns=["Ads", "Email", "Organic", "Referral"])
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    heatmap = ax.imshow(heatmap_data.values, cmap="YlOrRd", aspect="auto")

    midpoint = heatmap_data.values.mean()
    for row_index, segment in enumerate(heatmap_data.index):
        for col_index, channel in enumerate(heatmap_data.columns):
            value = heatmap_data.loc[segment, channel]
            text_color = "white" if value >= midpoint else "black"
            ax.text(
                col_index,
                row_index,
                f"{value:.1f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
            )

    ax.set_title("Figure 4. Churn Rate by Customer Segment and Marketing Channel", fontsize=14, pad=12)
    ax.set_xlabel("Marketing Channel")
    ax.set_ylabel("Customer Segment")
    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_xticklabels(heatmap_data.columns)
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index)
    plt.colorbar(heatmap, ax=ax, label="Churn Rate (%)")

    fig.tight_layout()
    output_path = OUTPUT_DIR / "figure_4_customer_segment_marketing_channel_heatmap.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    (BASE_DIR / ".matplotlib").mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(INPUT_FILE)

    output_files = [
        save_churn_rate_by_subcategory(df),
        save_average_churn_rate_by_age_group(df),
        save_churn_rate_by_delivery_time(df),
        save_churn_rate_heatmap_by_segment_channel(df),
    ]

    print("Generated four figures from the processed customer churn dataset:")
    for path in output_files:
        print(f"- {path.name}")


if __name__ == "__main__":
    main()
