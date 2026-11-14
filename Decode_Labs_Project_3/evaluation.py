
import numpy as np


# ==========================================
# 1. PRECISION AT K
# ==========================================

def precision_at_k(recommended_items, relevant_items, k=5):

    recommended = recommended_items[:k]

    if not recommended:
        return 0.0

    relevant_count = sum(
        1 for item in recommended
        if item in relevant_items
    )

    return relevant_count / len(recommended)


# ==========================================
# 2. RECALL AT K
# ==========================================

def recall_at_k(recommended_items, relevant_items, k=5):

    if not relevant_items:
        return 0.0

    recommended = recommended_items[:k]

    relevant_count = sum(
        1 for item in recommended
        if item in relevant_items
    )

    return relevant_count / len(relevant_items)


# ==========================================
# 3. F1 SCORE
# ==========================================

def f1_score(precision, recall):

    if precision + recall == 0:
        return 0.0

    return (
        2 * precision * recall
    ) / (precision + recall)


# ==========================================
# 4. ROOT MEAN SQUARED ERROR
# ==========================================

def calculate_rmse(actual_ratings, predicted_ratings):

    actual = np.array(actual_ratings, dtype=float)

    predicted = np.array(
        predicted_ratings, dtype=float
    )

    if len(actual) == 0 or len(actual) != len(predicted):
        raise ValueError(
            "Actual and predicted ratings must have "
            "equal, non-zero lengths."
        )

    errors = actual - predicted

    return float(np.sqrt(np.mean(errors ** 2)))


# ==========================================
# 5. CATALOGUE COVERAGE
# ==========================================

def catalogue_coverage(
    recommended_items,
    catalogue_items
):

    catalogue = set(catalogue_items)

    if not catalogue:
        return 0.0

    recommended = set(recommended_items)

    return len(
        recommended.intersection(catalogue)
    ) / len(catalogue)


# ==========================================
# 6. COMPLETE EVALUATION REPORT
# ==========================================

def evaluate_recommendations(
    recommended_items,
    relevant_items,
    catalogue_items,
    actual_ratings=None,
    predicted_ratings=None,
    k=5
):

    precision = precision_at_k(
        recommended_items,
        relevant_items,
        k
    )

    recall = recall_at_k(
        recommended_items,
        relevant_items,
        k
    )

    f1 = f1_score(
        precision,
        recall
    )

    coverage = catalogue_coverage(
        recommended_items,
        catalogue_items
    )

    report = {
        "Precision@K": precision,
        "Recall@K": recall,
        "F1 Score": f1,
        "Catalogue Coverage": coverage
    }

    # RMSE is only calculated when actual and
    # predicted ratings are both supplied.
    if actual_ratings is not None and predicted_ratings is not None:

        report["RMSE"] = calculate_rmse(
            actual_ratings,
            predicted_ratings
        )

    return report
