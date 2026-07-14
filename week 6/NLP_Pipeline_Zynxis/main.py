import pandas as pd
from sklearn.model_selection import train_test_split

from src.model import SentimentModel
from src.preprocessing import preprocess_text, tokenize


def main():
    df = pd.read_csv("data/feedback.csv")
    df["label"] = df["label"].str.lower().map({"positive": 1, "negative": 0})
    df["processed_text"] = df["text"].apply(preprocess_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["processed_text"],
        df["label"],
        test_size=0.25,
        random_state=42,
        stratify=df["label"],
    )

    sample_text = df.iloc[0]["text"]
    print("Original text:", sample_text)
    print("Tokenized text:", tokenize(sample_text))
    print("Processed text:", preprocess_text(sample_text))

    model = SentimentModel()
    model.fit(X_train, y_train)

    X_train_vec = model.vectorizer.transform(X_train)
    print("\nVectorized training shape:", X_train_vec.shape)

    accuracy, report = model.evaluate(X_test, y_test)
    print("\nAccuracy:", round(accuracy, 3))
    print("Classification report:\n", report)

    sample_inputs = [
        "The internship was amazing and the mentors were very supportive.",
        "I felt lost because the guidance was poor and the tasks were confusing."
    ]

    processed_samples = [preprocess_text(text) for text in sample_inputs]
    predicted_labels = model.predict(processed_samples)
    sentiment_map = {1: "positive", 0: "negative"}
    print("\nSample predictions:")
    for text, label in zip(sample_inputs, predicted_labels):
        print(f"- {text} -> {sentiment_map[int(label)]}")


if __name__ == "__main__":
    main()
