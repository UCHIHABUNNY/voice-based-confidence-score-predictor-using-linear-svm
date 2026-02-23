import joblib
import speech_recognition as sr
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model = joblib.load("svm_sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

recognizer = sr.Recognizer()

try:
    with sr.Microphone(device_index=1) as source:
        print("Adjusting for noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Speak now...")
        audio = recognizer.listen(source)

    text = recognizer.recognize_google(audio)
    print("You said:", text)

    # Vectorize
    text_vector = vectorizer.transform([text])

    # Predict
    prediction = model.predict(text_vector)[0]
    raw_score = model.decision_function(text_vector)[0]
    confidence_score = round((abs(raw_score) / (abs(raw_score) + 1)) * 100, 2)

    sentiment = "Positive" if prediction == 1 else "Negative"

    print("Sentiment:", sentiment)
    print("Confidence Score:", confidence_score, "%")

    print("Starting save process...")

    file_name = "confidence_results.csv"

    new_data = {
        "ID": 1,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Text": text,
        "Sentiment": sentiment,
        "Confidence Score (%)": confidence_score
    }

    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
        new_data["ID"] = len(df) + 1
    else:
        df = pd.DataFrame(columns=["ID", "Timestamp", "Text", "Sentiment", "Confidence Score (%)"])

    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)

    df.to_csv(file_name, index=False)

    print("Saved in structured format!")
    # -------- GRAPH SECTION --------
    df = pd.read_csv("confidence_results.csv")
    plt.figure()
    plt.plot(df["ID"], df["Confidence Score (%)"])
    plt.xlabel("Session ID")
    plt.ylabel("Confidence Score (%)")
    plt.title("Confidence Score Trend")
    plt.show()

except Exception as e:
    print("FULL ERROR:", e)