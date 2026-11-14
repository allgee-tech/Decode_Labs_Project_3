
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from database import load_ratings, save_rating

DATA_PATH = Path(__file__).parent / "data" / "books.csv"
FEEDBACK_PATH = Path(__file__).parent / "feedback.csv"


# ==========================================
# 1. LOAD THE BOOK CATALOGUE
# ==========================================

def load_books():

    df = pd.read_csv(DATA_PATH)

    df["genres_list"] = df["genres"].fillna("").apply(
        lambda value: [
            genre.strip()
            for genre in value.split(",")
            if genre.strip()
        ]
    )

    df["search_text"] = (
        df["title"].fillna("") + " " +
        df["author"].fillna("") + " " +
        df["genres"].fillna("") + " " +
        df["description"].fillna("")
    )

    return df


# ==========================================
# 2. LOAD USER FEEDBACK
# ==========================================

def load_feedback():
    return load_ratings()

# ==========================================
# 3. SAVE USER RATINGS
# ==========================================

def save_feedback(user_id, title, rating):
    save_rating(user_id, title, rating)


# ==========================================
# 4. CONTENT-BASED RECOMMENDATIONS
# ==========================================

def calculate_content_scores(
    books,
    interests,
    preferred_genres
):

    result = books.copy().reset_index(drop=True)

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    item_matrix = vectorizer.fit_transform(
        result["search_text"]
    )

    interest_text = " ".join(interests).strip()

    if interest_text:

        user_vector = vectorizer.transform(
            [interest_text]
        )

        text_scores = cosine_similarity(
            user_vector,
            item_matrix
        ).flatten()

    else:

        text_scores = np.zeros(len(result))

    preferred_genres = set(preferred_genres or [])

    def genre_match(genres):

        if not preferred_genres:
            return 0.0

        matches = set(genres).intersection(
            preferred_genres
        )

        return len(matches) / len(preferred_genres)

    genre_scores = result["genres_list"].apply(
        genre_match
    ).to_numpy()

    rating_scores = (
        result["rating"].fillna(0).clip(0, 5).to_numpy() / 5
    )

    min_year = result["year"].min()
    max_year = result["year"].max()

    if max_year == min_year:

        recency_scores = np.ones(len(result))

    else:

        recency_scores = (
            (result["year"] - min_year) /
            (max_year - min_year)
        ).to_numpy()

    return (
        result,
        item_matrix,
        text_scores,
        genre_scores,
        rating_scores,
        recency_scores
    )


# ==========================================
# 5. COLLABORATIVE FILTERING
# ==========================================

def collaborative_filtering(
    books,
    feedback,
    current_user
):

    # Start with a neutral score.
    scores = pd.Series(
        0.5,
        index=books.index,
        dtype=float
    )

    # Collaborative filtering needs ratings
    # from more than one user.
    if feedback["user_id"].nunique() < 2:

        return scores

    # Create a user-item matrix.
    # Rows = users
    # Columns = books
    # Values = ratings.
    user_item_matrix = feedback.pivot_table(
        index="user_id",
        columns="title",
        values="rating",
        aggfunc="mean"
    ).fillna(0)

    # Check whether the current user has rated books.
    if current_user not in user_item_matrix.index:

        return scores

    # Compare the current user with other users.
    similarity_matrix = cosine_similarity(
        user_item_matrix
    )

    user_position = list(
        user_item_matrix.index
    ).index(current_user)

    similarities = similarity_matrix[user_position]

    # Do not compare the user with themselves.
    similarities[user_position] = 0

    # Ignore users with zero similarity.
    similarities = np.maximum(similarities, 0)

    if similarities.sum() == 0:

        return scores

    # Predict ratings for books using similar users.
    for index, book in books.iterrows():

        title = book["title"]

        if title not in user_item_matrix.columns:
            continue

        book_ratings = user_item_matrix[title].to_numpy()

        # Only consider users who rated this book.
        rated_mask = book_ratings > 0

        # Exclude users with no similarity.
        useful_mask = rated_mask & (similarities > 0)

        if not useful_mask.any():
            continue

        relevant_ratings = book_ratings[useful_mask]

        relevant_weights = similarities[useful_mask]

        predicted_rating = np.average(
            relevant_ratings,
            weights=relevant_weights
        )

        # Convert the 1-5 rating into a 0-1 score.
        scores.loc[index] = predicted_rating / 5

    return scores.clip(0, 1)


# ==========================================
# 6. MAIN RECOMMENDATION ENGINE
# ==========================================

def recommend(
    books,
    interests,
    preferred_genres,
    min_rating=0.0,
    top_n=5,
    user_id="user_001"
):

    user_id = str(user_id).strip()

    (
        result,
        item_matrix,
        text_scores,
        genre_scores,
        rating_scores,
        recency_scores
    ) = calculate_content_scores(
        books,
        interests,
        preferred_genres
    )

    feedback = load_feedback()

    # Calculate collaborative filtering scores.
    collaborative_scores = collaborative_filtering(
        result,
        feedback,
        user_id
    ).to_numpy()

    # Calculate similarity to the current user's
    # positively and negatively rated books.
    personal_feedback = feedback[
        feedback["user_id"] == user_id
    ]

    feedback_scores = np.full(
        len(result), 0.5
    )

    if not personal_feedback.empty:

        positive_titles = personal_feedback[
            personal_feedback["rating"] >= 4
        ]["title"].tolist()

        negative_titles = personal_feedback[
            personal_feedback["rating"] <= 2
        ]["title"].tolist()

        title_positions = {
            title: position
            for position, title in enumerate(result["title"])
        }

        positive_indices = [
            title_positions[title]
            for title in positive_titles
            if title in title_positions
        ]

        negative_indices = [
            title_positions[title]
            for title in negative_titles
            if title in title_positions
        ]

        positive_similarity = np.zeros(len(result))
        negative_similarity = np.zeros(len(result))

        if positive_indices:

            positive_similarity = cosine_similarity(
                item_matrix,
                item_matrix[positive_indices]
            ).max(axis=1)

        if negative_indices:

            negative_similarity = cosine_similarity(
                item_matrix,
                item_matrix[negative_indices]
            ).max(axis=1)

        feedback_scores = np.clip(
            0.5
            + 0.5 * positive_similarity
            - 0.5 * negative_similarity,
            0,
            1
        )

    # Store individual recommendation scores.
    result["similarity_score"] = text_scores
    result["genre_score"] = genre_scores
    result["feedback_score"] = feedback_scores
    result["collaborative_score"] = collaborative_scores

    # HYBRID RECOMMENDATION FORMULA.
    result["recommendation_score"] = (

        0.45 * text_scores
        + 0.20 * genre_scores
        + 0.10 * rating_scores
        + 0.05 * recency_scores
        + 0.08 * feedback_scores
        + 0.12 * collaborative_scores

    ).clip(0, 1)

    # Do not recommend books the user has already rated.
    rated_titles = personal_feedback["title"].tolist()

    result = result[
        ~result["title"].isin(rated_titles)
    ]

    # Apply minimum catalogue rating.
    result = result[
        result["rating"] >= float(min_rating)
    ]

    # Sort from highest to lowest recommendation score.
    result = result.sort_values(
        by="recommendation_score",
        ascending=False
    )

    return result.head(int(top_n))
