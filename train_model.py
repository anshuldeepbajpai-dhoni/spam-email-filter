import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

from .config import DATA_PATH, MODEL_PATH
from .preprocess import clean_text


def main():
    print(f"Loading dataset from {DATA_PATH}")

    # Load your dataset
    df = pd.read_csv(DATA_PATH, encoding="latin-1")

    # Your dataset columns are EXACTLY: Category, Message
    df = df[["Category", "Message"]]
    df.columns = ["label", "message"]

    # Convert labels: ham = 0, spam = 1
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    # Ensure all messages are strings
    df["message"] = df["message"].astype(str)

    X = df["message"]
    y = df["label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # BEST MODEL FOR SMS SPAM — Linear SVM + TF-IDF
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=clean_text,
            ngram_range=(1, 2),
            min_df=1,
            max_features=20000
        )),
        ("clf", LinearSVC())
    ])

    print("\nTraining SVM spam classifier...")
    pipeline.fit(X_train, y_train)

    # Evaluate model
    preds = pipeline.predict(X_test)
    print("\nModel Accuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:\n", classification_report(y_test, preds))

    # Save model
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved at: {MODEL_PATH}")

    # ---------------------------------------------------------
    # TEST THE MODEL WITH A KNOWN SPAM MESSAGE (WAP message)
    # ---------------------------------------------------------
    print("\nTesting model with WAP message...")

    test_msg = "XXXMobileMovieClub: To use your credit, click the WAP link in the next txt message or click here"

    test_pred = pipeline.predict([test_msg])[0]

    if test_pred == 1:
        print("Prediction on test message: SPAM (CORRECT)")
    else:
        print("Prediction on test message: NOT SPAM (INCORRECT)")
    # ---------------------------------------------------------


if __name__ == "__main__":
    main()
