# SmartRecommend AI — DecodeLabs Project 3

A beginner-friendly but extensible content-based recommendation system built with Python and Streamlit.

## Project requirements covered

1. **Take user input:** users select genres, enter interests, choose a minimum rating and choose how many results to see.
2. **Match preferences:** the engine combines TF-IDF/cosine similarity, genre overlap and other weighted signals.
3. **Display recommendations:** books are ranked by match score with a plain-language explanation.

## Additional features

- Weighted recommendation score
- Explainable recommendations
- User rating capture saved to `feedback.csv`
- Recommendation analytics
- Adjustable filters
- Downloadable/editable CSV catalogue

## 1. Install Python

Use Python 3.10 or newer. Open this folder in VS Code.

## 2. Create a virtual environment

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

## 3. Install packages

```bash
python -m pip install -r requirements.txt
```

## 4. Run the application

```bash
streamlit run app.py
```

Streamlit will open the application in your browser. If it does not open automatically, visit the local URL shown in the terminal (usually `http://localhost:8501`).

## Project structure

```text
SmartRecommend_AI/
│
├── app.py                 # Streamlit user interface
├── recommender.py         # Recommendation and similarity logic
├── requirements.txt       # Required Python packages
├── README.md              # Setup and project guide
├── data/
│   └── books.csv          # Sample recommendation catalogue
└── feedback.csv           # Created automatically after saving a rating
```

## Understanding the algorithm

### TF-IDF
TF-IDF converts text into numerical vectors. Words that help distinguish one description from another receive more useful weights.

### Cosine similarity
Cosine similarity compares the angle between two vectors. A higher value means the text is more similar.

### Weighted score
The project combines four signals:

```text
Final score =
    0.60 × text similarity
  + 0.25 × genre match
  + 0.10 × catalogue rating
  + 0.05 × recency
  + feedback adjustment
```

The score is clamped between 0 and 1. The weights are project design choices and can be tested and improved.

## Important limitations and future improvements

- The sample catalogue is small and is provided for demonstration. Replace it with a larger, properly sourced dataset for a stronger evaluation.
- Saved ratings are stored locally in `feedback.csv`; this is not a multi-user database.
- The feedback adjustment currently changes the score of the rated book itself. It does not yet learn general preferences across all books in a genre.
- A future version can add user accounts, a database, collaborative filtering, evaluation metrics such as Precision@K and Recall@K, and model comparison.

## Suggested tests

1. Select Technology and Programming and enter "Python machine learning".
2. Select Finance and enter "saving money and investing".
3. Change the minimum rating and compare the results.
4. Rate a recommended book, refresh, and inspect the saved feedback.
5. Compare recommendations when you change only one preference at a time.
