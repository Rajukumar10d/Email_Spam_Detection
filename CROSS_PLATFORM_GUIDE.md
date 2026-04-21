# Cross-Platform Deployment Guide

This project already exports a lightweight model file at:

```text
model_artifacts/model_export.json
```

That export contains:

- the vocabulary
- class prior probabilities
- per-word log probabilities for ham and spam

## Why This Matters

`MultinomialNB` is a lightweight model. That makes it practical to reuse on:

- Android
- iPhone
- Windows desktop

without sending user messages to a cloud server.

## Practical Deployment Options

### 1. Use the same model inside each app

Workflow:

1. Train the model in Python.
2. Export the vocabulary and probabilities to JSON.
3. Bundle `model_export.json` inside each app.
4. Recreate the same text-cleaning and Naive Bayes scoring logic in:
   - Kotlin for Android
   - Swift for iOS
   - C# for Windows

This is the best option for an offline, privacy-first student project.

### 2. Use a backend API for all platforms

Workflow:

1. Train the model once.
2. Host prediction logic on a server.
3. Android, iOS, and Windows apps send text to the server.
4. The server returns spam or ham.

This is easier to maintain, but it is not offline and is weaker for privacy.

## Recommended Architecture For Your Project

For your novelty statement, use this:

> A lightweight offline spam detection engine that is trained in Python once and exported in a reusable format for Android, iOS, and Windows applications.

## What You Need To Match Across Platforms

Every platform must follow the same steps:

1. lowercase the message
2. replace links with `linktoken`
3. remove special characters
4. split into tokens
5. count token frequency
6. look up token indexes in the vocabulary
7. apply Naive Bayes scoring using the exported probabilities
8. return:
   - predicted label
   - spam probability
   - top spam indicators

## Platform Mapping

### Android

- language: Kotlin
- package the JSON file in app assets
- load it at app startup
- run prediction fully offline

### iPhone

- language: Swift
- package the JSON file in the app bundle
- load it locally
- run prediction fully offline

### Windows

- language: C# with WinForms, WPF, or .NET MAUI
- include the JSON file as app content
- run local inference without internet

## Important Real-World Limitation

Your model can be embedded inside apps on all these platforms, but operating systems usually do not allow your app to silently read messages from every installed app without permission.

So the realistic project scope is:

- detect spam inside your own app
- classify messages shared into your app
- connect to a permitted message or email source

## Demo-Friendly Explanation

You can explain it like this:

> The model is trained in Python, but because Naive Bayes is mathematically lightweight, the learned vocabulary and probabilities can be exported and reused inside Android, iOS, and Windows applications for offline spam detection.
