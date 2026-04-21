import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "SMSSpamCollection"


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " linktoken ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download the UCI SMS Spam Collection and place "
            "the extracted SMSSpamCollection file inside the data folder."
        )

    encodings_to_try = ["utf-8", "latin-1", "cp1252"]
    last_error = None
    records = None

    for encoding in encodings_to_try:
        try:
            records = []
            with path.open("r", encoding=encoding) as dataset_file:
                for line_number, raw_line in enumerate(dataset_file, start=1):
                    line = raw_line.rstrip("\r\n")
                    if not line:
                        continue

                    parts = line.split("\t", maxsplit=1)
                    if len(parts) != 2:
                        raise ValueError(
                            f"Expected label and text at line {line_number}, found: {line!r}"
                        )

                    label, text = parts
                    records.append({"label": label, "text": text})
            break
        except UnicodeDecodeError as error:
            last_error = error
            records = None

    if records is None:
        raise UnicodeDecodeError(
            last_error.encoding,
            last_error.object,
            last_error.start,
            last_error.end,
            "Unable to decode dataset with supported encodings.",
        )

    df = pd.DataFrame(records)

    df["clean_text"] = df["text"].apply(clean_text)
    df["target"] = df["label"].map({"ham": 0, "spam": 1})
    return df


def build_model() -> Pipeline:
    return Pipeline(
        [
            ("vectorizer", CountVectorizer(stop_words="english")),
            ("classifier", MultinomialNB()),
        ]
    )


def train_model(df: pd.DataFrame) -> Pipeline:
    model = build_model()
    model.fit(df["clean_text"], df["target"])
    return model


def explain_prediction(model: Pipeline, message: str, top_n: int = 5) -> dict:
    cleaned = clean_text(message)
    vectorizer = model.named_steps["vectorizer"]
    classifier = model.named_steps["classifier"]

    transformed = vectorizer.transform([cleaned])
    prediction = model.predict([cleaned])[0]
    probabilities = model.predict_proba([cleaned])[0]

    feature_names = vectorizer.get_feature_names_out()
    token_counts = Counter(cleaned.split())
    token_scores = []

    for token, count in token_counts.items():
        index = vectorizer.vocabulary_.get(token)
        if index is None:
            continue

        ham_log_prob = classifier.feature_log_prob_[0][index]
        spam_log_prob = classifier.feature_log_prob_[1][index]
        token_scores.append(
            {
                "token": token,
                "count": count,
                "spam_weight": round(float(spam_log_prob - ham_log_prob), 4),
            }
        )

    top_tokens = sorted(token_scores, key=lambda item: item["spam_weight"], reverse=True)[:top_n]
    return {
        "cleaned_text": cleaned,
        "predicted_label": "spam" if prediction == 1 else "ham",
        "spam_probability": round(float(probabilities[1]), 4),
        "ham_probability": round(float(probabilities[0]), 4),
        "matched_features": int(transformed.nnz),
        "top_spam_indicators": top_tokens,
        "vocabulary_size": len(feature_names),
    }


def print_prediction_summary(message: str, explanation: dict) -> None:
    print("\nMessage:", message)
    print("Prediction:", explanation["predicted_label"].upper())
    print(
        "Confidence:",
        f"spam={explanation['spam_probability']:.4f}, ham={explanation['ham_probability']:.4f}",
    )
    print("Matched features:", explanation["matched_features"])
    print("Top spam indicators:")
    if explanation["top_spam_indicators"]:
        for indicator in explanation["top_spam_indicators"]:
            print(
                f"  - {indicator['token']} (count={indicator['count']}, "
                f"spam_weight={indicator['spam_weight']:.4f})"
            )
    else:
        print("  - No strong spam indicators were matched in the vocabulary.")


def run_interactive_demo(model: Pipeline) -> None:
    print("\nInteractive mode:")
    print("Type a message and press Enter to classify it.")
    print("Type 'quit' to stop.\n")

    while True:
        user_message = input("Enter message: ").strip()
        if user_message.lower() in {"quit", "exit"}:
            print("Interactive demo closed.")
            break
        if not user_message:
            print("Please enter a message.\n")
            continue

        explanation = explain_prediction(model, user_message)
        print_prediction_summary(user_message, explanation)
        print()


def print_sample_predictions(model: Pipeline, messages: Iterable[str]) -> None:
    print("\nExplainable sample predictions:")
    for message in messages:
        explanation = explain_prediction(model, message)
        print_prediction_summary(message, explanation)


def main() -> None:
    df = load_dataset(DATASET_PATH)

    print("\nDataset preview:")
    print(df[["label", "text"]].head())

    print("\nClass distribution:")
    print(df["label"].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["target"],
        test_size=0.2,
        random_state=42,
        stratify=df["target"],
    )

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nEvaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    sample_messages = [
        "Congratulations! You have won a free entry to our prize draw. Click now.",
        "Hi, are we still meeting after class today?",
        "URGENT! Your mobile number has won cash. Reply immediately.",
    ]

    print_sample_predictions(model, sample_messages)

    run_interactive_demo(model)


if __name__ == "__main__":
    main()
