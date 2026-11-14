
from evaluation import evaluate_recommendations


# Books recommended by our AI.
recommended_books = [
    "Python Crash Course",
    "Clean Code",
    "Deep Work",
    "Hands-On Machine Learning",
    "The Lean Startup"
]

# Books considered relevant in this test.
relevant_books = [
    "Python Crash Course",
    "Hands-On Machine Learning",
    "The Pragmatic Programmer"
]

# All books available in the catalogue.
catalogue = [
    "Python Crash Course",
    "Clean Code",
    "Deep Work",
    "Hands-On Machine Learning",
    "The Lean Startup",
    "The Pragmatic Programmer",
    "Atomic Habits"
]

# Example held-out ratings and predictions.
actual_ratings = [5, 4, 3]
predicted_ratings = [4.5, 4, 3.5]


results = evaluate_recommendations(
    recommended_items=recommended_books,
    relevant_items=relevant_books,
    catalogue_items=catalogue,
    actual_ratings=actual_ratings,
    predicted_ratings=predicted_ratings,
    k=5
)


print("\nSMARTRECOMMEND AI - EVALUATION REPORT")
print("=" * 45)

for metric, value in results.items():

    print(f"{metric}: {value:.4f}")

print("=" * 45)
