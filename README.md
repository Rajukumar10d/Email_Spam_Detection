# Satark Sandesh

Satark Sandesh is an Indian-themed spam detection web app that checks whether a message looks safe or suspicious using NLP and Naive Bayes.

This project detects spam messages automatically using NLP and a Naive Bayes model trained on the **UCI SMS Spam Collection** dataset.

It is designed as a strong student project with these goals:

- train and evaluate a working spam detector
- explain why a message was predicted as spam
- present the model in a simple, user-friendly web app
- export the model in a lightweight JSON-friendly form for future Android, iPhone, and Windows integration

## Dataset

- Official UCI dataset page: [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- Direct download link: [Download zip](https://cdn.uci-ics-mlr-prod.aws.uci.edu/228/smsspamcollection.zip)

After downloading, extract the file named `SMSSpamCollection` into the `data/` folder so the final path is:

```text
data/SMSSpamCollection
```

## Project Structure

```text
.
|-- data/
|-- model_artifacts/
|-- CROSS_PLATFORM_GUIDE.md
|-- app.py
|-- export_model.py
|-- gui_app.py
|-- requirements.txt
|-- spam_detector.py
|-- static/
|-- templates/
|-- README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Training

```bash
python spam_detector.py
```

This script will:

- load the dataset
- preprocess text with simple NLP cleaning
- train a `CountVectorizer + MultinomialNB` pipeline
- evaluate accuracy, precision, recall, and confusion matrix
- show explainable predictions on sample messages
- open an interactive console mode where you can test your own messages

## Export Lightweight Model Data

```bash
python export_model.py
```

This creates `model_artifacts/model_export.json`.

That JSON is useful because later you can reuse the trained vocabulary and Naive Bayes probabilities in:

- Android with Kotlin
- iPhone with Swift
- Windows with C#/.NET

## Run The Interactive Demo

```bash
python spam_detector.py
```

After the evaluation section, type your own message and the program will show:

- predicted label
- spam and ham confidence
- matched features
- top spam indicators

Type `quit` to exit the interactive mode.

## Run The GUI Demo

```bash
python gui_app.py
```

This opens a simple desktop interface where you can paste a message and see:

- spam or ham result
- spam probability
- top indicator words

## Run The Responsive Web App

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

This version gives you:

- a modern responsive interface
- mobile-friendly layout
- live prediction through a Flask API
- a cleaner presentation for project demos

## Cross-Platform Reuse

See [CROSS_PLATFORM_GUIDE.md](CROSS_PLATFORM_GUIDE.md) for the Android, iPhone, and Windows deployment idea using `model_export.json`.

## Why This Project Is More Than a Basic Classifier

This project is built around a stronger novelty statement:

> A lightweight, offline, explainable spam detection engine that can be reused across Android, iOS, and Windows applications.

## Future Improvements

- replace `CountVectorizer` with `TfidfVectorizer`
- add stopword removal and stemming
- detect phishing indicators such as suspicious links and urgent scam language
- support Hinglish or multilingual inputs
- use user feedback to improve the model over time
