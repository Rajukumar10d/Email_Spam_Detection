import json
from pathlib import Path

from spam_detector import DATASET_PATH, build_model, load_dataset


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "model_artifacts"
OUTPUT_PATH = OUTPUT_DIR / "model_export.json"


def main() -> None:
    df = load_dataset(DATASET_PATH)
    model = build_model()
    model.fit(df["clean_text"], df["target"])

    vectorizer = model.named_steps["vectorizer"]
    classifier = model.named_steps["classifier"]

    export_data = {
        "labels": ["ham", "spam"],
        "vocabulary": vectorizer.vocabulary_,
        "class_log_prior": classifier.class_log_prior_.tolist(),
        "feature_log_prob": classifier.feature_log_prob_.tolist(),
        "metadata": {
            "dataset": "UCI SMS Spam Collection",
            "vectorizer": "CountVectorizer(stop_words='english')",
            "classifier": "MultinomialNB",
            "note": (
                "This exported data can be reused in a lightweight inference engine "
                "for Android, iOS, or Windows apps."
            ),
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(export_data, indent=2), encoding="utf-8")

    print(f"Model exported to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
