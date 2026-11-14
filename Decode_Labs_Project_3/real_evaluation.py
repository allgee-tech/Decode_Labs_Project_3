
import pandas as pd
import numpy as np

import recommender
from database import load_ratings
from evaluation import (
    precision_at_k,
    recall_at_k,
    f1_score,
    calculate_rmse,
    catalogue_coverage
)


# ==========================================
# 1. LOAD THE DATABASE AND BOOK CATALOGUE
# ==========================================

books = recommender.load_books()
ratings = load_ratings()

K = 5

print("\nSMARTRECOMMEND AI")
print("REAL-DATA MODEL EVALUATION")
print("=" * 50)

if ratings.empty:
    print("No user ratings found in the database.")
    raise SystemExit

print(f"Total ratings: {len(ratings)}")
print(f"Total users: {ratings['user_id'].nunique()}")
print(f"Total books rated: {ratings['title'].nunique()}")


# ==========================================
# 2. PREPARE EVALUATION VARIABLES
# ==========================================

all_recommendations = []
all_relevant_books = []
all_catalogue_books = books["title"].tolist()

actual_ratings = []
predicted_ratings = []

evaluated_users = 0
skipped_users = 0

original_load_feedback = recommender.load_feedback


# ==========================================
# 3. LEAVE-ONE-OUT EVALUATION
# ==========================================

try:

    for user_id in ratings["user_id"].unique():

        user_ratings = ratings[
            ratings["user_id"] == user_id
        ].copy()

        # A user needs at least two ratings:
        # one for testing and at least one for training.
        if len(user_ratings) < 2:

            skipped_users += 1
            continue

        # Select a positively rated book for testing.
        positive_ratings = user_ratings[
            user_ratings["rating"] >= 4
        ]

        if positive_ratings.empty:

            skipped_users += 1
            continue

        # Hold out one positive rating.
        test_rating = positive_ratings.iloc[0]

        hidden_title = test_rating["title"]
        actual_rating = float(test_rating["rating"])

        # Remove the hidden rating from training data.
        training_ratings = ratings[
            ~(
                (ratings["user_id"] == user_id) &
                (ratings["title"] == hidden_title)
            )
        ].copy()

        # Temporarily give the recommendation engine
        # only the training ratings.
        recommender.load_feedback = (
            lambda training=training_ratings: training.copy()
        )

        # Generate recommendations.
        recommendations = recommender.recommend(
            books=books,
            interests=[],
            preferred_genres=[],
            min_rating=0,
            top_n=K,
            user_id=user_id
        )

        recommended_titles = recommendations[
            "title"
        ].tolist()

        # Record the evaluation results.
        all_recommendations.append(
            recommended_titles
        )

        all_relevant_books.append(
            {hidden_title}
        )

        # Calculate collaborative rating prediction.
        training_feedback = training_ratings

        collaborative_scores = (
            recommender.collaborative_filtering(
                books.reset_index(drop=True),
                training_feedback,
                user_id
            )
        )

        matching_books = books[
            books["title"] == hidden_title
        ].reset_index(drop=True)

        if not matching_books.empty:

            matching_position = books[
                "title"
            ].reset_index(drop=True).tolist().index(
                hidden_title
            )

            predicted_score = float(
                collaborative_scores.iloc[
                    matching_position
                ]
            )

            # Convert the 0-1 score to a 1-5
            # rating scale.
            predicted_rating = 1 + (
                predicted_score * 4
            )

            actual_ratings.append(actual_rating)
            predicted_ratings.append(predicted_rating)

        evaluated_users += 1

finally:

    # Restore the original feedback function.
    recommender.load_feedback = original_load_feedback


# ==========================================
# 4. CALCULATE FINAL METRICS
# ==========================================

print("\nEVALUATION SUMMARY")
print("-" * 50)

print(f"Users evaluated: {evaluated_users}")
print(f"Users skipped: {skipped_users}")

if evaluated_users == 0:

    print(
        "\nNot enough rating data to evaluate the model."
    )

    print(
        "Each evaluated user needs at least two ratings, "
        "including one rating of 4 or 5."
    )

    raise SystemExit


# Aggregate precision and recall across users.
# Each user has one held-out relevant book.
precision_values = []
recall_values = []

for recommendations, relevant in zip(
    all_recommendations,
    all_relevant_books
):

    precision_values.append(
        precision_at_k(recommendations, relevant, K)
    )

    recall_values.append(
        recall_at_k(recommendations, relevant, K)
    )


precision = float(np.mean(precision_values))
recall = float(np.mean(recall_values))
f1 = f1_score(precision, recall)

coverage = catalogue_coverage(
    all_recommendations,
    all_catalogue_books
)

print(f"Precision@{K}: {precision:.4f}")
print(f"Recall@{K}: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Catalogue Coverage: {coverage:.4f}")

if actual_ratings:

    rmse = calculate_rmse(
        actual_ratings,
        predicted_ratings
    )

    print(f"RMSE: {rmse:.4f}")

else:

    print("RMSE: Not available")

print("=" * 50)
print("REAL-DATA EVALUATION COMPLETED")


# ==========================================
# 5. SAVE EVALUATION RESULTS
# ==========================================

report = {
    "Precision@K": precision,
    "Recall@K": recall,
    "F1 Score": f1,
    "Catalogue Coverage": coverage,
    "Users Evaluated": evaluated_users,
    "Users Skipped": skipped_users,
    "Total Ratings": len(ratings),
    "Total Users": ratings["user_id"].nunique()
}

if actual_ratings:
    report["RMSE"] = calculate_rmse(
        actual_ratings,
        predicted_ratings
    )

# Save the report for the dashboard.
report_df = pd.DataFrame([report])

report_df.to_csv(
    "evaluation_results.csv",
    index=False
)

print("\nEvaluation results saved successfully.")
print("File: evaluation_results.csv")

