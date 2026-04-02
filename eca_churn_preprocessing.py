from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "ECA_churn.csv"
CLEANED_FILE = BASE_DIR / "ECA_churn_cleaned.csv"
FEATURES_FILE = BASE_DIR / "ECA_churn_features.csv"
TARGET_FILE = BASE_DIR / "ECA_churn_target.csv"
STRUCTURAL_MISSING_SEGMENT_COUNTRIES = {"Switzerland", "France", "Australia"}

RENAME_MAP = {
    "Customer ID": "customer_id",
    "StockCode": "stock_code",
    "Quantity": "quantity",
    "Price": "price",
    "Country": "country",
    "Customer_Age": "customer_age",
    "Gender": "gender",
    "Customer_Segment": "customer_segment",
    "Marketing_Channel": "marketing_channel",
    "Category": "category",
    "Subcategory": "subcategory",
    "Discount_Applied": "discount_applied",
    "Payment_Method": "payment_method",
    "Delivery_Time_Days": "delivery_time_days",
    "Churn_Flag": "churn_flag",
}


def count_iqr_outliers(series: pd.Series) -> tuple[int, float, float]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_count = int(((series < lower) | (series > upper)).sum())
    return outlier_count, float(lower), float(upper)


def cap_iqr_outliers(series: pd.Series) -> tuple[pd.Series, dict[str, float]]:
    outlier_count, lower, upper = count_iqr_outliers(series)
    capped = series.clip(lower=lower, upper=upper)
    return capped, {
        "outliers_before": outlier_count,
        "lower_cap": lower,
        "upper_cap": upper,
    }


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Technique: Rename variables to Python-friendly snake_case names.
    return df.rename(columns=RENAME_MAP)


def check_duplicate_data(df: pd.DataFrame) -> dict[str, int | bool]:
    exact_duplicate_rows = int(df.duplicated().sum())
    customer_id_duplicates = int(df["customer_id"].duplicated().sum())
    return {
        "has_exact_duplicate_rows": exact_duplicate_rows > 0,
        "exact_duplicate_row_count": exact_duplicate_rows,
        "has_duplicate_customer_ids": customer_id_duplicates > 0,
        "duplicate_customer_id_count": customer_id_duplicates,
    }


def remove_duplicate_entries(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    # Technique: Drop repeated rows while keeping the first occurrence of each record.
    deduplicated_df = df.drop_duplicates(keep="first").reset_index(drop=True)

    report = {
        "rows_before_deduplication": int(df.shape[0]),
        "duplicate_rows_before": int(df.duplicated().sum()),
        "duplicate_rows_removed": int(df.shape[0] - deduplicated_df.shape[0]),
        "rows_after_deduplication": int(deduplicated_df.shape[0]),
    }
    return deduplicated_df, report


def summarize_missing_values(df: pd.DataFrame) -> dict[str, int]:
    return {
        column: int(count)
        for column, count in df.isna().sum().items()
    }


def summarize_country_missing_labels(df: pd.DataFrame) -> dict[str, int]:
    labeled_counts = (
        df.loc[
            df["country"].isin(STRUCTURAL_MISSING_SEGMENT_COUNTRIES)
            & df["customer_segment"].eq("Missing"),
            "country",
        ]
        .value_counts()
        .reindex(sorted(STRUCTURAL_MISSING_SEGMENT_COUNTRIES), fill_value=0)
    )
    return {country: int(count) for country, count in labeled_counts.items()}


def impute_missing_numeric_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    df = df.copy()
    report: dict[str, dict[str, float]] = {}

    # Technique: Replace missing numeric values with the mean of the non-missing values.
    numeric_columns = [
        column
        for column in df.select_dtypes(include=["number"]).columns.tolist()
        if column != "customer_id"
    ]

    for column in numeric_columns:
        missing_before = int(df[column].isna().sum())
        if missing_before == 0:
            continue

        mean_value = float(df[column].mean())
        df[column] = df[column].fillna(mean_value)
        report[column] = {
            "missing_before": missing_before,
            "imputation_value": mean_value,
        }

    return df, report


def clean_categorical_values(
    df: pd.DataFrame,
    stock_code_min_frequency: int = 3,
) -> tuple[pd.DataFrame, dict[str, dict[str, str | int]]]:
    df = df.copy()
    report: dict[str, dict[str, str | int]] = {}

    # Technique: Treat categorical values by trimming and standardizing text fields.
    string_columns = [
        "stock_code",
        "country",
        "gender",
        "customer_segment",
        "marketing_channel",
        "category",
        "subcategory",
        "discount_applied",
        "payment_method",
        "churn_flag",
    ]

    for column in string_columns:
        df[column] = df[column].astype("string").str.strip()

    # Technique: Treat placeholder and missing categorical values using a
    # country-aware rule before fallback mode imputation.
    customer_segment_missing_question = int(df["customer_segment"].eq("Missing?").sum())
    df["customer_segment"] = df["customer_segment"].replace({"Missing?": pd.NA})
    customer_segment_missing_before = int(df["customer_segment"].isna().sum())
    structural_missing_mask = (
        df["country"].isin(STRUCTURAL_MISSING_SEGMENT_COUNTRIES)
        & df["customer_segment"].isna()
    )
    structural_missing_filled = int(structural_missing_mask.sum())
    df.loc[structural_missing_mask, "customer_segment"] = "Missing"
    customer_segment_missing_after_country_rule = int(df["customer_segment"].isna().sum())
    customer_segment_mode = str(df["customer_segment"].mode(dropna=True).iloc[0])
    df["customer_segment"] = df["customer_segment"].fillna(customer_segment_mode)
    report["customer_segment"] = {
        "missing_question_before": customer_segment_missing_question,
        "missing_before": customer_segment_missing_before,
        "country_missing_labeled": structural_missing_filled,
        "missing_after_country_rule": customer_segment_missing_after_country_rule,
        "mode_imputed": customer_segment_mode,
    }

    # Technique: Reduce number of categories by merging inconsistent labels.
    df["category"] = df["category"].replace({"Home_Decor": "Home Decor"})

    # Technique: Treat categorical values by standardizing binary/target text labels.
    df["discount_applied"] = df["discount_applied"].str.lower()
    df["churn_flag"] = df["churn_flag"].str.lower()

    # Technique: Reduce number of categories by grouping rare stock_code values into "Other".
    stock_code_counts = df["stock_code"].value_counts()
    keep_codes = stock_code_counts[stock_code_counts >= stock_code_min_frequency].index
    df["stock_code"] = df["stock_code"].where(df["stock_code"].isin(keep_codes), "Other")

    return df, report


def treat_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    df = df.copy()
    report: dict[str, dict[str, float]] = {}

    # Technique: Treat outliers in numerical values using IQR-based capping.
    for column in ["quantity", "price"]:
        df[column], report[column] = cap_iqr_outliers(df[column])

    df["price"] = df["price"].round(2)
    return df, report


def build_model_inputs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    # Technique: Extract dependent variable (y) from the churn flag.
    y = df["churn_flag"].map({"active": 0, "churned": 1})
    if y.isna().any():
        invalid = sorted(df.loc[y.isna(), "churn_flag"].dropna().unique().tolist())
        raise ValueError(f"Unexpected churn_flag values found: {invalid}")

    # Technique: Extract independent variables (X) by removing the target and identifier.
    X = df.drop(columns=["churn_flag", "customer_id"]).copy()

    # Technique: Treat categorical/binary values by converting yes/no into 1/0.
    X["discount_applied"] = X["discount_applied"].map({"no": 0, "yes": 1})

    # Technique: Create dummy variables for the remaining categorical predictors.
    categorical_columns = X.select_dtypes(include=["string", "object"]).columns.tolist()
    X_encoded = pd.get_dummies(X, columns=categorical_columns, dtype=int)

    return X_encoded, y.astype("int64")


def main() -> None:
    raw_df = load_data(INPUT_FILE)

    assessment = {
        "rows": int(raw_df.shape[0]),
        "columns": int(raw_df.shape[1]),
        "total_missing_values_before": int(raw_df.isna().sum().sum()),
        "missing_customer_segment_before": int(raw_df["Customer_Segment"].isna().sum()),
        "missing_question_customer_segment_before": int(raw_df["Customer_Segment"].astype("string").eq("Missing?").sum()),
        "category_levels_before": int(raw_df["Category"].nunique(dropna=False)),
        "stock_code_levels_before": int(raw_df["StockCode"].nunique(dropna=False)),
    }
    assessment["quantity_outliers_before"], _, _ = count_iqr_outliers(raw_df["Quantity"])
    assessment["price_outliers_before"], _, _ = count_iqr_outliers(raw_df["Price"])

    renamed_df = rename_columns(raw_df)
    duplicate_check_before = check_duplicate_data(renamed_df)
    deduplicated_df, duplicate_report = remove_duplicate_entries(renamed_df)
    imputed_df, missing_value_report = impute_missing_numeric_values(deduplicated_df)
    cleaned_df, categorical_report = clean_categorical_values(imputed_df)
    cleaned_df, outlier_report = treat_outliers(cleaned_df)
    duplicate_check_after = check_duplicate_data(cleaned_df)
    cleaned_missing_summary = summarize_missing_values(cleaned_df)
    country_missing_label_summary = summarize_country_missing_labels(cleaned_df)
    X, y = build_model_inputs(cleaned_df)

    cleaned_df.to_csv(CLEANED_FILE, index=False)
    X.to_csv(FEATURES_FILE, index=False)
    pd.DataFrame({"churn_flag": y}).to_csv(TARGET_FILE, index=False)

    print("ECA churn preprocessing completed.")
    print()
    print("Selected preprocessing tasks:")
    print("1. Rename variables to Python-friendly snake_case.")
    print("2. Remove duplicate rows while keeping all unique records.")
    print("3. Replace missing numeric values with the column mean.")
    print("4. Replace missing customer_segment values with 'Missing' for countries where the segment is fully absent.")
    print("5. Replace the remaining missing categorical values with the column mode.")
    print("6. Treat numeric outliers before modeling.")
    print()
    print("Duplicate handling summary:")
    for key, value in duplicate_report.items():
        print(f"- {key}: {value}")
    print()
    print("Duplicate check summary:")
    for key, value in duplicate_check_before.items():
        print(f"- raw_{key}: {value}")
    for key, value in duplicate_check_after.items():
        print(f"- cleaned_{key}: {value}")
    print()
    print("Missing value treatment summary:")
    if missing_value_report:
        for column, details in missing_value_report.items():
            print(
                f"- {column}: missing_before={int(details['missing_before'])} | "
                f"mean_imputed={details['imputation_value']:.4f}"
            )
    else:
        print("- No numeric columns required mean imputation.")
    print()
    print("Categorical treatment summary:")
    print(f"- country_rule_countries: {sorted(STRUCTURAL_MISSING_SEGMENT_COUNTRIES)}")
    for country, count in country_missing_label_summary.items():
        print(f"- {country}: Missing for all {count} affected rows")
    for column, details in categorical_report.items():
        print(
            f"- {column}: missing_question_before={int(details['missing_question_before'])} | "
            f"missing_before={int(details['missing_before'])} | "
            f"country_missing_labeled={int(details['country_missing_labeled'])} | "
            f"missing_after_country_rule={int(details['missing_after_country_rule'])} | "
            f"mode_imputed={details['mode_imputed']}"
        )
    print()
    print("Assessment before preprocessing:")
    for key, value in assessment.items():
        print(f"- {key}: {value}")
    print()
    print("Categorical cleaning summary:")
    print(f"- missing_customer_segment_after: {int(cleaned_df['customer_segment'].isna().sum())}")
    print(f"- category_levels_after: {int(cleaned_df['category'].nunique(dropna=False))}")
    print(f"- stock_code_levels_after: {int(cleaned_df['stock_code'].nunique(dropna=False))}")
    print(f"- customer_segment_levels: {sorted(cleaned_df['customer_segment'].dropna().unique().tolist())}")
    print()
    print("Missing values by column after cleaning:")
    for column, count in cleaned_missing_summary.items():
        print(f"- {column}: {count}")
    print()
    print("Outlier treatment summary:")
    for column, details in outlier_report.items():
        after_count, _, _ = count_iqr_outliers(cleaned_df[column])
        print(
            f"- {column}: capped using IQR "
            f"[{details['lower_cap']:.2f}, {details['upper_cap']:.2f}] | "
            f"outliers_before={int(details['outliers_before'])} | outliers_after={after_count}"
        )
    print()
    print("Model-ready output:")
    print(f"- cleaned_dataset: {CLEANED_FILE.name} -> {cleaned_df.shape}")
    print(f"- feature_matrix: {FEATURES_FILE.name} -> {X.shape}")
    print(f"- target_vector: {TARGET_FILE.name} -> {(y.shape[0], 1)}")
    print(f"- churn_rate: {y.mean():.4f}")


if __name__ == "__main__":
    main()
