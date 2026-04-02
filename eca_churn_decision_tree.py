from __future__ import annotations

import os
import textwrap
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split, validation_curve
from sklearn.tree import DecisionTreeClassifier, plot_tree


BASE_DIR = SCRIPT_DIR
FEATURES_FILE = BASE_DIR / "ECA_churn_features.csv"
TARGET_FILE = BASE_DIR / "ECA_churn_target.csv"
OUTPUT_DIR = BASE_DIR / "decision_tree_outputs"
CATEGORICAL_FEATURE_PREFIXES = [
    "stock_code",
    "country",
    "gender",
    "customer_segment",
    "marketing_channel",
    "category",
    "subcategory",
    "payment_method",
]


def load_processed_data() -> tuple[pd.DataFrame, pd.Series]:
    # Step 1: Load the independent variables (X) and dependent variable (y)
    # that were created from the processed churn dataset in Question 1.
    X = pd.read_csv(FEATURES_FILE)
    y = pd.read_csv(TARGET_FILE)["churn_flag"]
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # Step 2: Split the data into training and test sets so the model can be
    # trained on one subset and evaluated on unseen data.
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )


def train_baseline_tree(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
    # Step 3a: Train an unconstrained baseline tree to expose how a fully grown
    # tree behaves before regularization.
    baseline_model = DecisionTreeClassifier(random_state=42, class_weight="balanced")
    baseline_model.fit(X_train, y_train)
    return baseline_model


def train_decision_tree(X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    # Step 3b: Tune the decision tree using cross-validation and regularization
    # settings while keeping the final tuned tree at a fixed depth of 4.
    base_model = DecisionTreeClassifier(random_state=42, class_weight="balanced")
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [4],
        "min_samples_leaf": [1, 2, 5, 10],
        "min_samples_split": [2, 5, 10, 20, 40],
        "ccp_alpha": [0.0, 0.0005, 0.001, 0.002],
    }

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=1,
        return_train_score=True,
    )
    grid_search.fit(X_train, y_train)
    return grid_search


def select_balanced_model(
    grid_search: GridSearchCV,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tolerance: float = 0.08,
    preferred_actual_depth: int = 4,
) -> tuple[DecisionTreeClassifier, dict[str, object], pd.DataFrame]:
    # Step 3c: Choose a near-best cross-validation model with a smaller
    # train-validation gap while keeping a fuller, interpretable depth-4 tree.
    cv_results = pd.DataFrame(grid_search.cv_results_).copy()
    cv_results["gap"] = cv_results["mean_train_score"] - cv_results["mean_test_score"]

    candidate_pool = cv_results[cv_results["mean_test_score"] >= grid_search.best_score_ - tolerance].copy()
    candidate_pool["param_max_depth_numeric"] = candidate_pool["param_max_depth"].astype(int)
    candidate_pool["param_min_samples_leaf_numeric"] = candidate_pool["param_min_samples_leaf"].astype(int)
    candidate_pool["param_min_samples_split_numeric"] = candidate_pool["param_min_samples_split"].astype(int)
    candidate_pool["param_ccp_alpha_numeric"] = candidate_pool["param_ccp_alpha"].astype(float)
    candidate_models: list[DecisionTreeClassifier] = []
    actual_depths: list[int] = []
    leaf_counts: list[int] = []

    for _, row in candidate_pool.iterrows():
        candidate_model = DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced",
            criterion=str(row["param_criterion"]),
            max_depth=int(row["param_max_depth"]),
            min_samples_leaf=int(row["param_min_samples_leaf"]),
            min_samples_split=int(row["param_min_samples_split"]),
            ccp_alpha=float(row["param_ccp_alpha"]),
        )
        candidate_model.fit(X_train, y_train)
        candidate_models.append(candidate_model)
        actual_depths.append(candidate_model.get_depth())
        leaf_counts.append(candidate_model.get_n_leaves())

    candidate_pool = candidate_pool.reset_index(drop=True)
    candidate_pool["actual_depth"] = actual_depths
    candidate_pool["leaf_count"] = leaf_counts
    candidate_pool["depth_distance"] = (candidate_pool["actual_depth"] - preferred_actual_depth).abs()

    preferred_candidates = candidate_pool[candidate_pool["actual_depth"] == preferred_actual_depth].copy()
    if preferred_candidates.empty:
        preferred_candidates = candidate_pool.copy()

    selected_row = preferred_candidates.sort_values(
        by=[
            "depth_distance",
            "leaf_count",
            "mean_test_score",
            "gap",
            "param_ccp_alpha_numeric",
            "param_min_samples_leaf_numeric",
            "param_min_samples_split_numeric",
        ],
        ascending=[True, False, False, True, True, True, True],
    ).iloc[0]

    selected_params = {
        "criterion": str(selected_row["param_criterion"]),
        "max_depth": int(selected_row["param_max_depth"]),
        "min_samples_leaf": int(selected_row["param_min_samples_leaf"]),
        "min_samples_split": int(selected_row["param_min_samples_split"]),
        "ccp_alpha": float(selected_row["param_ccp_alpha"]),
        "cv_mean_test_score": float(selected_row["mean_test_score"]),
        "cv_mean_train_score": float(selected_row["mean_train_score"]),
        "cv_gap": float(selected_row["gap"]),
        "actual_depth": int(selected_row["actual_depth"]),
        "leaf_count": int(selected_row["leaf_count"]),
    }

    selected_model = candidate_models[int(selected_row.name)]
    return selected_model, selected_params, cv_results


def evaluate_model(
    model: DecisionTreeClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, object]:
    # Step 4: Evaluate the fitted decision tree on both training and test data.
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)

    metrics = {
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "train_precision": precision_score(y_train, y_train_pred, zero_division=0),
        "train_recall": recall_score(y_train, y_train_pred, zero_division=0),
        "train_f1": train_f1,
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "test_precision": precision_score(y_test, y_test_pred, zero_division=0),
        "test_recall": recall_score(y_test, y_test_pred, zero_division=0),
        "test_f1": test_f1,
        "f1_gap": train_f1 - test_f1,
        "tree_depth": model.get_depth(),
        "leaf_count": model.get_n_leaves(),
        "confusion_matrix": confusion_matrix(y_test, y_test_pred),
        "classification_report": classification_report(
            y_test,
            y_test_pred,
            target_names=["active", "churned"],
            zero_division=0,
        ),
    }
    return metrics


def save_model_comparison(
    baseline_metrics: dict[str, object],
    tuned_metrics: dict[str, object],
    output_path: Path,
) -> pd.DataFrame:
    # Step 5a: Compare an unconstrained baseline tree with the tuned tree to show
    # how regularization reduces overfitting.
    comparison_df = pd.DataFrame(
        [
            {
                "model": "baseline_unconstrained_tree",
                "tree_depth": baseline_metrics["tree_depth"],
                "leaf_count": baseline_metrics["leaf_count"],
                "train_accuracy": baseline_metrics["train_accuracy"],
                "test_accuracy": baseline_metrics["test_accuracy"],
                "train_f1": baseline_metrics["train_f1"],
                "test_f1": baseline_metrics["test_f1"],
                "f1_gap": baseline_metrics["f1_gap"],
            },
            {
                "model": "tuned_regularized_tree",
                "tree_depth": tuned_metrics["tree_depth"],
                "leaf_count": tuned_metrics["leaf_count"],
                "train_accuracy": tuned_metrics["train_accuracy"],
                "test_accuracy": tuned_metrics["test_accuracy"],
                "train_f1": tuned_metrics["train_f1"],
                "test_f1": tuned_metrics["test_f1"],
                "f1_gap": tuned_metrics["f1_gap"],
            },
        ]
    )
    comparison_df.to_csv(output_path, index=False)
    return comparison_df


def save_complexity_curve(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    selected_params: dict[str, object],
    output_path: Path,
) -> None:
    # Step 5b: Plot training vs validation F1 across tree depths to visualize
    # the trade-off between underfitting and overfitting.
    depth_values = [1, 2, 3, 4, 5, 6, 8, 10, 12]
    curve_model = DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced",
        criterion=selected_params["criterion"],
        min_samples_leaf=selected_params["min_samples_leaf"],
        min_samples_split=selected_params["min_samples_split"],
        ccp_alpha=selected_params["ccp_alpha"],
    )

    train_scores, validation_scores = validation_curve(
        curve_model,
        X_train,
        y_train,
        param_name="max_depth",
        param_range=depth_values,
        scoring="f1",
        cv=5,
        n_jobs=1,
    )

    train_mean = train_scores.mean(axis=1)
    validation_mean = validation_scores.mean(axis=1)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(depth_values, train_mean, marker="o", linewidth=2, color="#2A6F97", label="Training F1")
    ax.plot(depth_values, validation_mean, marker="o", linewidth=2, color="#D1495B", label="Validation F1")
    ax.axvline(selected_params["max_depth"], linestyle="--", color="#444444", label="Selected max_depth")

    ax.set_title("Decision Tree Complexity Curve", fontsize=14, pad=12)
    ax.set_xlabel("Max Depth")
    ax.set_ylabel("F1-Score")
    ax.set_xticks(depth_values)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_cv_results(grid_search: GridSearchCV, output_path: Path) -> pd.DataFrame:
    # Step 5c: Save the cross-validation results to document the model selection process.
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results["gap"] = cv_results["mean_train_score"] - cv_results["mean_test_score"]
    cv_results = cv_results[
        [
            "rank_test_score",
            "mean_test_score",
            "mean_train_score",
            "gap",
            "std_test_score",
            "param_criterion",
            "param_max_depth",
            "param_min_samples_leaf",
            "param_min_samples_split",
            "param_ccp_alpha",
        ]
    ].sort_values(["rank_test_score", "mean_test_score"], ascending=[True, False])
    cv_results.to_csv(output_path, index=False)
    return cv_results


def save_confusion_matrix_plot(conf_matrix, output_path: Path) -> None:
    # Step 6a: Save a confusion matrix plot to visualize correct and incorrect predictions.
    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["active", "churned"])
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Decision Tree Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def format_tree_feature_name(feature_name: str, wrap_width: int = 18) -> str:
    for prefix in CATEGORICAL_FEATURE_PREFIXES:
        marker = f"{prefix}_"
        if feature_name.startswith(marker):
            category = prefix.replace("_", " ").title()
            value = feature_name[len(marker) :].replace("_", " ")
            return textwrap.fill(f"{category} = {value}", width=wrap_width)

    return textwrap.fill(feature_name.replace("_", " ").title(), width=wrap_width)


def save_tree_plot(model: DecisionTreeClassifier, feature_names: list[str], output_path: Path) -> None:
    # Step 6b: Save the decision tree structure so the split rules can be interpreted.
    formatted_feature_names = [format_tree_feature_name(name) for name in feature_names]
    full_level_slots = 2 ** model.get_depth()
    figure_width = min(max(16, full_level_slots * 2.8), 30)
    figure_height = min(max(8, model.get_depth() * 2.8 + 2), 16)
    font_size = 10 if model.get_depth() <= 5 else 9

    fig, ax = plt.subplots(figsize=(figure_width, figure_height), facecolor="#F7F3EC")
    ax.set_facecolor("#F7F3EC")
    annotations = plot_tree(
        model,
        feature_names=formatted_feature_names,
        class_names=["active", "churned"],
        filled=True,
        rounded=True,
        impurity=False,
        proportion=True,
        precision=2,
        fontsize=font_size,
        ax=ax,
    )
    for annotation in annotations:
        annotation.set_color("#1F2933")
        annotation.set_fontfamily("DejaVu Sans")
        annotation.set_linespacing(1.15)
        bbox = annotation.get_bbox_patch()
        if bbox is not None:
            bbox.set_edgecolor("#5C4B51")
            bbox.set_linewidth(1.2)
            bbox.set_alpha(0.98)

    ax.set_title("Decision Tree Model for Customer Churn", fontsize=16, color="#5C4B51", pad=18)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_feature_importance(model: DecisionTreeClassifier, feature_names: list[str], output_path: Path) -> pd.DataFrame:
    # Step 6c: Save the most important predictors used by the decision tree.
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance_df.to_csv(output_path, index=False)
    return importance_df


def save_metrics_summary(
    grid_search: GridSearchCV,
    selected_params: dict[str, object],
    baseline_metrics: dict[str, object],
    tuned_metrics: dict[str, object],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    top_features: pd.DataFrame,
    comparison_df: pd.DataFrame,
    output_path: Path,
) -> None:
    # Step 7: Save a plain-language summary of the decision tree construction and results.
    top_10 = top_features.head(10)

    with output_path.open("w", encoding="utf-8") as file:
        file.write("Decision Tree Model for Customer Churn\n")
        file.write("=" * 40 + "\n\n")

        file.write("Relevant steps used to construct the model:\n")
        file.write("1. Load the processed feature matrix and churn target from Question 1.\n")
        file.write("2. Split the data into training and test sets using stratified sampling.\n")
        file.write("3. Fit an unconstrained baseline tree to check how much a fully grown tree overfits.\n")
        file.write("4. Tune the decision tree hyperparameters with 5-fold cross-validation.\n")
        file.write("5. Limit model complexity using max_depth, min_samples_split, min_samples_leaf, and ccp_alpha pruning.\n")
        file.write("6. Choose a balanced model from the near-best CV candidates while preferring enough depth to interpret the tree.\n")
        file.write("7. Interpret the final trained model using the plotted tree and feature importances.\n\n")

        file.write("Efforts used to prevent overfitting and underfitting:\n")
        file.write("- Used a stratified train/test split to evaluate the model on unseen data.\n")
        file.write("- Compared a baseline unconstrained tree against a tuned regularized tree.\n")
        file.write("- Tuned max_depth to stop the tree from growing too deep.\n")
        file.write("- Tuned min_samples_split and min_samples_leaf to avoid very small unstable splits.\n")
        file.write("- Tuned ccp_alpha to apply post-pruning and remove weak branches.\n")
        file.write("- Used 5-fold cross-validation and a complexity curve to balance bias and variance.\n")
        file.write("- Selected the final model from the near-best CV candidates using a balance of validation score, depth, and generalization gap.\n\n")

        file.write("Dataset summary:\n")
        file.write(f"- Total observations: {len(y_train) + len(y_test)}\n")
        file.write(f"- Total features: {X_train.shape[1]}\n")
        file.write(f"- Training set size: {len(y_train)}\n")
        file.write(f"- Test set size: {len(y_test)}\n\n")

        file.write("Best hyperparameters from cross-validation:\n")
        for key, value in grid_search.best_params_.items():
            file.write(f"- {key}: {value}\n")
        file.write(f"- Best cross-validation F1-score: {grid_search.best_score_:.4f}\n\n")

        file.write("Final selected balanced-model hyperparameters:\n")
        file.write(f"- criterion: {selected_params['criterion']}\n")
        file.write(f"- max_depth: {selected_params['max_depth']}\n")
        file.write(f"- min_samples_leaf: {selected_params['min_samples_leaf']}\n")
        file.write(f"- min_samples_split: {selected_params['min_samples_split']}\n")
        file.write(f"- ccp_alpha: {selected_params['ccp_alpha']}\n")
        file.write(f"- CV mean training F1: {selected_params['cv_mean_train_score']:.4f}\n")
        file.write(f"- CV mean validation F1: {selected_params['cv_mean_test_score']:.4f}\n")
        file.write(f"- CV train-validation gap: {selected_params['cv_gap']:.4f}\n")
        file.write(f"- Actual fitted depth: {selected_params['actual_depth']}\n")
        file.write(f"- Actual leaf count: {selected_params['leaf_count']}\n\n")

        file.write("Baseline vs tuned model comparison:\n")
        for _, row in comparison_df.iterrows():
            file.write(
                f"- {row['model']}: depth={int(row['tree_depth'])}, leaves={int(row['leaf_count'])}, "
                f"train_f1={row['train_f1']:.4f}, test_f1={row['test_f1']:.4f}, f1_gap={row['f1_gap']:.4f}\n"
            )
        file.write("\n")

        if baseline_metrics["f1_gap"] > tuned_metrics["f1_gap"]:
            file.write(
                "Interpretation: the tuned tree reduced the train-test F1 gap, "
                f"from {baseline_metrics['f1_gap']:.4f} to {tuned_metrics['f1_gap']:.4f}, "
                "which is evidence of reduced overfitting.\n\n"
            )
        else:
            file.write(
                "Interpretation: the tuned tree did not reduce the train-test F1 gap, "
                "so the complexity settings may need further adjustment.\n\n"
            )

        file.write("Final tuned model performance:\n")
        file.write(f"- Training accuracy: {tuned_metrics['train_accuracy']:.4f}\n")
        file.write(f"- Training precision: {tuned_metrics['train_precision']:.4f}\n")
        file.write(f"- Training recall: {tuned_metrics['train_recall']:.4f}\n")
        file.write(f"- Training F1-score: {tuned_metrics['train_f1']:.4f}\n")
        file.write(f"- Test accuracy: {tuned_metrics['test_accuracy']:.4f}\n")
        file.write(f"- Test precision: {tuned_metrics['test_precision']:.4f}\n")
        file.write(f"- Test recall: {tuned_metrics['test_recall']:.4f}\n")
        file.write(f"- Test F1-score: {tuned_metrics['test_f1']:.4f}\n")
        file.write(f"- Train-test F1 gap: {tuned_metrics['f1_gap']:.4f}\n\n")

        file.write("Classification report:\n")
        file.write(str(tuned_metrics["classification_report"]))
        file.write("\n")

        file.write("Top 10 most important features:\n")
        for _, row in top_10.iterrows():
            file.write(f"- {row['feature']}: {row['importance']:.4f}\n")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    (BASE_DIR / ".matplotlib").mkdir(exist_ok=True)

    X, y = load_processed_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    baseline_model = train_baseline_tree(X_train, y_train)
    grid_search = train_decision_tree(X_train, y_train)
    selected_model, selected_params, _ = select_balanced_model(grid_search, X_train, y_train)

    baseline_metrics = evaluate_model(baseline_model, X_train, X_test, y_train, y_test)
    tuned_metrics = evaluate_model(selected_model, X_train, X_test, y_train, y_test)

    comparison_path = OUTPUT_DIR / "decision_tree_model_comparison.csv"
    complexity_curve_path = OUTPUT_DIR / "decision_tree_complexity_curve.png"
    cv_results_path = OUTPUT_DIR / "decision_tree_cv_results.csv"
    confusion_matrix_path = OUTPUT_DIR / "decision_tree_confusion_matrix.png"
    tree_plot_path = OUTPUT_DIR / "decision_tree_model.png"
    feature_importance_path = OUTPUT_DIR / "decision_tree_feature_importance.csv"
    metrics_summary_path = OUTPUT_DIR / "decision_tree_metrics.txt"

    comparison_df = save_model_comparison(baseline_metrics, tuned_metrics, comparison_path)
    save_complexity_curve(X_train, y_train, selected_params, complexity_curve_path)
    save_cv_results(grid_search, cv_results_path)
    save_confusion_matrix_plot(tuned_metrics["confusion_matrix"], confusion_matrix_path)
    top_features = save_feature_importance(selected_model, X.columns.tolist(), feature_importance_path)
    save_tree_plot(selected_model, X.columns.tolist(), tree_plot_path)
    save_metrics_summary(
        grid_search,
        selected_params,
        baseline_metrics,
        tuned_metrics,
        X_train,
        X_test,
        y_train,
        y_test,
        top_features,
        comparison_df,
        metrics_summary_path,
    )

    print("Decision tree modeling completed.")
    print(f"- Best parameters: {grid_search.best_params_}")
    print(f"- Best cross-validation F1-score: {grid_search.best_score_:.4f}")
    print(f"- Selected balanced-model parameters: {selected_params}")
    print(f"- Baseline tree train/test F1 gap: {baseline_metrics['f1_gap']:.4f}")
    print(f"- Tuned tree train/test F1 gap: {tuned_metrics['f1_gap']:.4f}")
    print(f"- Test accuracy: {tuned_metrics['test_accuracy']:.4f}")
    print(f"- Test precision: {tuned_metrics['test_precision']:.4f}")
    print(f"- Test recall: {tuned_metrics['test_recall']:.4f}")
    print(f"- Test F1-score: {tuned_metrics['test_f1']:.4f}")
    print(f"- Outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
