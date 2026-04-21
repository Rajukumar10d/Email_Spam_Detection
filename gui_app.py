import tkinter as tk
from tkinter import messagebox, scrolledtext

from spam_detector import explain_prediction, load_dataset, train_model, DATASET_PATH


class SpamDetectorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Spam Detector Demo")
        self.root.geometry("760x620")
        self.root.configure(bg="#f4f7fb")

        dataset = load_dataset(DATASET_PATH)
        self.model = train_model(dataset)

        self.build_ui()

    def build_ui(self) -> None:
        title = tk.Label(
            self.root,
            text="Offline Spam Detection Demo",
            font=("Segoe UI", 20, "bold"),
            bg="#f4f7fb",
            fg="#12344d",
        )
        title.pack(pady=(20, 6))

        subtitle = tk.Label(
            self.root,
            text="Enter a message to classify it as spam or ham with explainable indicators.",
            font=("Segoe UI", 10),
            bg="#f4f7fb",
            fg="#486581",
        )
        subtitle.pack(pady=(0, 16))

        self.input_box = scrolledtext.ScrolledText(
            self.root,
            wrap=tk.WORD,
            width=85,
            height=10,
            font=("Segoe UI", 11),
        )
        self.input_box.pack(padx=20, pady=8)

        button_frame = tk.Frame(self.root, bg="#f4f7fb")
        button_frame.pack(pady=10)

        detect_button = tk.Button(
            button_frame,
            text="Detect Spam",
            font=("Segoe UI", 11, "bold"),
            bg="#1f6feb",
            fg="white",
            activebackground="#1558b0",
            activeforeground="white",
            padx=18,
            pady=8,
            command=self.detect_message,
        )
        detect_button.pack(side=tk.LEFT, padx=6)

        clear_button = tk.Button(
            button_frame,
            text="Clear",
            font=("Segoe UI", 11),
            bg="#d9e2ec",
            fg="#102a43",
            padx=18,
            pady=8,
            command=self.clear_all,
        )
        clear_button.pack(side=tk.LEFT, padx=6)

        self.result_label = tk.Label(
            self.root,
            text="Prediction will appear here.",
            font=("Segoe UI", 14, "bold"),
            bg="#f4f7fb",
            fg="#102a43",
        )
        self.result_label.pack(pady=(14, 8))

        self.detail_box = scrolledtext.ScrolledText(
            self.root,
            wrap=tk.WORD,
            width=85,
            height=14,
            font=("Consolas", 10),
            state=tk.DISABLED,
        )
        self.detail_box.pack(padx=20, pady=(0, 20))

    def detect_message(self) -> None:
        message = self.input_box.get("1.0", tk.END).strip()
        if not message:
            messagebox.showwarning("Input needed", "Please enter a message to classify.")
            return

        explanation = explain_prediction(self.model, message)
        predicted_label = explanation["predicted_label"].upper()
        spam_probability = explanation["spam_probability"]
        color = "#b42318" if predicted_label == "SPAM" else "#027a48"

        self.result_label.config(
            text=f"Prediction: {predicted_label} | Spam probability: {spam_probability:.4f}",
            fg=color,
        )

        details = [
            f"Cleaned text: {explanation['cleaned_text']}",
            f"Matched features: {explanation['matched_features']}",
            f"Ham probability: {explanation['ham_probability']:.4f}",
            f"Spam probability: {explanation['spam_probability']:.4f}",
            "",
            "Top spam indicators:",
        ]

        indicators = explanation["top_spam_indicators"]
        if indicators:
            for item in indicators:
                details.append(
                    f"- {item['token']} | count={item['count']} | spam_weight={item['spam_weight']:.4f}"
                )
        else:
            details.append("- No strong indicators matched the learned vocabulary.")

        self.detail_box.config(state=tk.NORMAL)
        self.detail_box.delete("1.0", tk.END)
        self.detail_box.insert(tk.END, "\n".join(details))
        self.detail_box.config(state=tk.DISABLED)

    def clear_all(self) -> None:
        self.input_box.delete("1.0", tk.END)
        self.result_label.config(text="Prediction will appear here.", fg="#102a43")
        self.detail_box.config(state=tk.NORMAL)
        self.detail_box.delete("1.0", tk.END)
        self.detail_box.config(state=tk.DISABLED)


def main() -> None:
    root = tk.Tk()
    app = SpamDetectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
