 Voice-Based Confidence Score Predictor (Linear SVM)
This project analyzes speaker confidence by converting real-time speech into text and applying a Supervised Machine Learning model to classify the speaker's state as "Confident" or "Not Confident".

 Key Features
Real-time Transcription: Uses the Google Speech Recognition API to convert audio input into processed text.

Machine Learning Engine: Features a Linear SVM (Support Vector Machine) classifier trained for binary classification.

Advanced NLP: Implements TF-IDF Vectorization with an n-gram range (1, 3) to capture deep linguistic context and phrases.

Optimization: Enhanced through Hyperparameter Tuning (optimizing the C-parameter) to ensure high model generalization and predictive accuracy.

Scoring Logic: Provides a probability-based confidence score and categorizes results into Low, Medium, or High Confidence levels.

 Technical Stack
Language: Python

Libraries: Scikit-learn, Pandas, Joblib, SpeechRecognition

Model: Linear SVM (Binary Classifier)

Vectorization: TF-IDF (Unigrams, Bigrams, Trigrams)

 Performance & Methodology
Initially designed as a rule-based sentiment mapping system using TextBlob, the project was upgraded to a data-driven approach to improve precision. By training on labeled speech data, the model achieved an estimated 85% accuracy in identifying confidence indicators within spoken text.

 Project Structure
train_model.py: Script for loading data, vectorizing text, and training the SVM model.

record.py: Main inference script for real-time speech capture and confidence prediction.

confidence_model.pkl: The trained and serialized machine learning model.
