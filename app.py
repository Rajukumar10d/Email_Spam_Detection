from pathlib import Path

from flask import Flask, jsonify, render_template, request

from spam_detector import DATASET_PATH, explain_prediction, load_dataset, train_model


BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, template_folder="templates", static_folder="static")

_dataset = load_dataset(DATASET_PATH)
_model = train_model(_dataset)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    if not message:
        return jsonify({"error": "Please enter a message to analyze."}), 400

    explanation = explain_prediction(_model, message)
    risk_level = (
        "High"
        if explanation["spam_probability"] >= 0.8
        else "Medium"
        if explanation["spam_probability"] >= 0.4
        else "Low"
    )

    response = {
        "message": message,
        "prediction": explanation["predicted_label"],
        "spam_probability": explanation["spam_probability"],
        "ham_probability": explanation["ham_probability"],
        "matched_features": explanation["matched_features"],
        "risk_level": risk_level,
        "cleaned_text": explanation["cleaned_text"],
        "top_spam_indicators": explanation["top_spam_indicators"],
    }
    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
