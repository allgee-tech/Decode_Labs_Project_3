import streamlit as st
import pandas as pd
from recommender import load_books, recommend, save_feedback, load_feedback

from database import (
    initialize_database,
    register_user,
    authenticate_user,
    get_user_ratings
)

initialize_database()

st.set_page_config(
    page_title="SmartRecommend AI",
    page_icon="✨",
    layout="centered"
)


# ==========================================
# USER AUTHENTICATION
# ==========================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = None


# Display login and registration when logged out.
if not st.session_state.logged_in:

    st.title("✨ SmartRecommend AI")

    login_tab, register_tab = st.tabs([
        "Login",
        "Create New Account"
    ])

    # LOGIN SECTION
    with login_tab:

        st.subheader("Welcome back")

        with st.form("login_form"):

            login_username = st.text_input("Username")

            login_password = st.text_input(
                "Password",
                type="password"
            )

            login_button = st.form_submit_button(
                "Login",
                use_container_width=True
            )

            if login_button:

                if authenticate_user(
                    login_username,
                    login_password
                ):

                    st.session_state.logged_in = True

                    st.session_state.username = (
                        login_username.strip()
                    )

                    st.rerun()

                else:

                    st.error(
                        "Invalid username or password."
                    )

    # REGISTRATION SECTION
    with register_tab:

        st.subheader("Create your account")

        with st.form("registration_form"):

            new_username = st.text_input(
                "Choose a username"
            )

            new_email = st.text_input(
                "Email address"
            )

            new_password = st.text_input(
                "Create password",
                type="password"
            )

            confirm_password = st.text_input(
                "Confirm password",
                type="password"
            )

            register_button = st.form_submit_button(
                "Create Account",
                use_container_width=True
            )

            if register_button:

                if new_password != confirm_password:

                    st.error("Passwords do not match.")

                else:

                    success, message = register_user(
                        new_username,
                        new_email,
                        new_password
                    )

                    if success:

                        st.success(message)

                    else:

                        st.error(message)

    st.stop()


# ==========================================
# LOGGED-IN USER INFORMATION
# ==========================================

user_id = st.session_state.username

st.sidebar.success(
    f"Logged in as {user_id}"
)

if st.sidebar.button("Logout"):

    st.session_state.logged_in = False
    st.session_state.username = None

    st.rerun()


# Simple custom styling for a clean dashboard.
st.markdown("""
<style>
    .main {background-color: #f7f8fc;}
    .hero {
        padding: 1.5rem 1.8rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #21134d, #3757a6);
        color: white;
        margin-bottom: 1.2rem;
    }
    .small-muted {color: #687087; font-size: 0.9rem;}
    div[data-testid="stMetric"] {
        background: white; padding: 14px; border-radius: 12px;
        border: 1px solid #e8eaf2;
    }
</style>
""", unsafe_allow_html=True)

books = load_books()
all_genres = sorted({genre for group in books["genres_list"] for genre in group})

st.markdown("""
<div class="hero">
  <h1>✨ SmartRecommend AI</h1>
  <p>Discover books that match your interests through explainable recommendation logic.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar: collect user preferences.
st.sidebar.header("Your preferences")


user_id = user_id.strip()

if not user_id:
    st.warning("Please enter a User ID.")
    st.stop()

st.sidebar.write(
    "Tell the system what you are interested in."
)

selected_genres = st.sidebar.multiselect(
    "Choose your preferred genres",
    options=all_genres,
    default=["Technology", "Programming"]
)
interest_text = st.sidebar.text_area(
    "What topics interest you?",
    placeholder="Example: Python, artificial intelligence, business, personal growth..."
)
min_rating = st.sidebar.slider("Minimum book rating", 0.0, 5.0, 3.5, 0.5)
top_n = st.sidebar.slider("Number of recommendations", 3, 10, 5)

interests = [word.strip() for word in interest_text.split(",") if word.strip()]
if not interests:
    interests = selected_genres

results = recommend(
    books=books,
    interests=interests,
    preferred_genres=selected_genres,
    min_rating=min_rating,
    top_n=top_n,
    user_id=user_id
)

# Dashboard metrics.
c1, c2, c3 = st.columns(3)
c1.metric("Books in catalogue", len(books))
c2.metric("Available genres", len(all_genres))
c3.metric("Recommendations found", len(results))

st.subheader("Your recommendations")
st.caption("Scores combine text similarity, genre match, catalogue rating and publication recency.")

if results.empty:
    st.warning("No books match those filters. Try lowering the minimum rating or selecting more genres.")
else:
    for _, book in results.iterrows():
        with st.container(border=True):
            left, right = st.columns([4, 1])
            with left:
                st.markdown(f"### {book['title']}")
                st.write(f"**Author:** {book['author']}  ·  **Year:** {int(book['year'])}")
                st.write(f"**Genres:** {book['genres']}")
                st.write(book["description"])
                # Explain the recommendation in plain language.
                reasons = []
                if book["genre_score"] > 0:
                    reasons.append("matches your selected genres")
                if book["similarity_score"] > 0.05:
                    reasons.append("contains topics related to your interests")
                if book["rating"] >= 4.5:
                    reasons.append("has a strong catalogue rating")
                if not reasons:
                    reasons.append("has the strongest available combined score")
                st.caption("Why this was recommended: " + "; ".join(reasons) + ".")
            with right:
                st.metric("Match score", f"{book['recommendation_score'] * 100:.0f}%")
                st.write(f"⭐ {book['rating']}/5")
                user_rating = st.selectbox(
                    "Your rating",
                    options=[1, 2, 3, 4, 5],
                    index=4,
                    key=f"rating_{book['title']}"
                )
                if st.button("Save rating", key=f"save_{book['title']}"):
                    save_feedback(
                        user_id=user_id,
                        title=book["title"],
                        rating=user_rating  
                    )
                    st.success("Rating saved!")

st.divider()
with st.expander("Recommendation analytics"):
    if not results.empty:
        chart_data = results.set_index("title")[["recommendation_score"]].copy()
        chart_data["recommendation_score"] *= 100
        st.bar_chart(chart_data)
    feedback = load_feedback()
    st.write(f"Saved user ratings: **{len(feedback)}**")
    if not feedback.empty:
        st.dataframe(feedback, use_container_width=True)

with st.expander("How the recommendation engine works"):
    st.markdown("""
    The system uses a weighted scoring model:

    - **60% Text similarity:** TF-IDF represents words numerically and cosine similarity compares your interests with each book.
    - **25% Genre match:** Measures overlap between your selected genres and a book's genres.
    - **10% Catalogue rating:** Gives a small advantage to highly rated books.
    - **5% Recency:** Gives a small advantage to newer books.
    - **Feedback adjustment:** Previously saved ratings can slightly adjust the score of the rated book.

    These weights are educational design choices for this project, not values specified by DecodeLabs.
    """)

# ==========================================
# PERSONAL RATING HISTORY
# ==========================================

with st.expander("My Rating History"):

    history = get_user_ratings(user_id)

    if history.empty:

        st.info(
            "You haven't rated any books yet."
        )

    else:

        st.dataframe(
            history,
            use_container_width=True
        )    

st.caption("DecodeLabs Industrial Training · Project 3 · AI Recommendation Logic")
