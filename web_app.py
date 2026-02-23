import streamlit as st
import joblib
import speech_recognition as sr
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt

# -------- PAGE CONFIG --------
st.set_page_config(page_title="Voice Confidence Analyzer", layout="centered")

st.title("🎙 Voice Confidence Analyzer")
st.markdown("Analyze your speaking confidence using AI")

# -------- LOAD MODEL --------
model = joblib.load("svm_sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# -------- RECORD BUTTON --------
if st.button("🎤 Start Recording"):

    recognizer = sr.Recognizer()

    with sr.Microphone(device_index=1) as source:
        st.info("Adjusting for noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        st.success("Speak now...")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        st.write("### 🗣 You Said:")
        st.write(text)

        # Vectorize
        text_vector = vectorizer.transform([text])

        # Predict
        prediction = model.predict(text_vector)[0]
        raw_score = model.decision_function(text_vector)[0]
        confidence_score = round((abs(raw_score) / (abs(raw_score) + 1)) * 100, 2)

        sentiment = "Positive" if prediction == 1 else "Negative"

        st.write("### 📊 Sentiment:", sentiment)
        st.write("### 🔥 Confidence Score:", f"{confidence_score} %")

        # -------- SAVE TO CSV --------
        file_name = "confidence_results.csv"

        new_data = {
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
            new_data["ID"] = 1

        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
        df = df[["ID", "Timestamp", "Text", "Sentiment", "Confidence Score (%)"]]
        df.to_csv(file_name, index=False)

        st.success("Saved Successfully!")

    except Exception as e:
        st.error(f"Error: {e}")

# -------- SHOW DATA --------
if os.path.exists("confidence_results.csv"):
    st.subheader("📁 Previous Records")
    df = pd.read_csv("confidence_results.csv")
    st.dataframe(df)

    st.subheader("📈 Confidence Trend")
    st.line_chart(df["Confidence Score (%)"])