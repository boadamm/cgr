import streamlit as st
import joblib
import tensorflow as tf
import numpy as np
from keras.models import load_model
from transformers import AutoTokenizer, TFDistilBertForSequenceClassification
from keras.preprocessing.sequence import pad_sequences
from utils.preprocess import clean_text
import pandas as pd
import matplotlib.pyplot as plt

# --- Load Models ---
try:
    logreg = joblib.load("outputs/logreg.pkl")
    rf = joblib.load("outputs/ranfor.pkl")
    nb = joblib.load("outputs/naivebayes.pkl")
    
    lstm_model = load_model("outputs/lstm_model.h5")
    tokenizer_lstm = joblib.load("outputs/lstm_tokenizer.pkl")
    
    bert_model = TFDistilBertForSequenceClassification.from_pretrained("outputs/bert_model/")
    bert_tokenizer = AutoTokenizer.from_pretrained("outputs/bert_model/")
    
    vectorizer = joblib.load("outputs/tfidf_vectorizer.pkl")
    # Verify vectorizer is properly fitted
    if not hasattr(vectorizer, 'vocabulary_') or not hasattr(vectorizer, 'idf_'):
        st.error("Error: TF-IDF vectorizer is not properly fitted. Please retrain the model.")
        st.stop()
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

# --- App UI ---
st.set_page_config(page_title="Fake Review Detector", layout="centered")
st.title("🕵️‍♂️ Multi-Model Fake Review Checker")
st.markdown("Enter a product review below. Each model will predict whether it's **Fake (CG)** or **Original (OR)**.")

review = st.text_area("✍️ Review Input:")

# --- Persistent Stats ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- Prediction Logic ---
if st.button("🔍 Check Review") and review.strip():
    cleaned = clean_text(review)

    # Traditional models
    X_input = vectorizer.transform([cleaned])
    logreg_pred = logreg.predict(X_input)[0]
    rf_pred = rf.predict(X_input)[0]
    nb_pred = nb.predict(X_input)[0]

    # LSTM
    seq = tokenizer_lstm.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=100)
    lstm_prob = lstm_model.predict(padded)[0][0]
    lstm_pred = int(lstm_prob > 0.5)

    # BERT
    bert_inputs = bert_tokenizer(cleaned, return_tensors="tf", truncation=True, padding=True)
    bert_logits = bert_model(bert_inputs).logits
    bert_probs = tf.nn.softmax(bert_logits, axis=1).numpy()[0]
    bert_pred = int(np.argmax(bert_probs))

    # Aggregate
    preds = {
        "Logistic Regression": logreg_pred,
        "Random Forest": rf_pred,
        "Naive Bayes": nb_pred,
        "LSTM": lstm_pred,
        "DistilBERT": bert_pred
    }

    probs = {
        "LSTM": lstm_prob,
        "DistilBERT Fake": bert_probs[1],
        "DistilBERT Original": bert_probs[0]
    }

    # Majority vote
    votes = list(preds.values())
    majority_vote = int(sum(votes) > len(votes) / 2)
    majority_label = "🟥 Fake" if majority_vote == 1 else "🟩 Original"

    st.subheader("🧠 Model Predictions")
    for model, pred in preds.items():
        label = "🟥 Fake" if pred == 1 else "🟩 Original"
        st.markdown(f"- **{model}**: {label}")

    st.markdown("---")
    st.subheader("📊 Prediction Probabilities")
    st.markdown(f"- **LSTM Probability** (Fake): `{lstm_prob:.2f}`")
    st.markdown(f"- **DistilBERT Probabilities** → Fake: `{bert_probs[1]:.2f}` | Original: `{bert_probs[0]:.2f}`")

    # Consensus
    st.markdown("---")
    st.subheader("🤝 Consensus")
    st.markdown(f"**Majority Vote Decision:** {majority_label}")

    # Save to history
    st.session_state.history.append({
        "text": cleaned,
        "LogReg": logreg_pred,
        "RF": rf_pred,
        "NB": nb_pred,
        "LSTM": lstm_pred,
        "BERT": bert_pred,
        "Consensus": majority_vote
    })

    # Statistics
    history_df = pd.DataFrame(st.session_state.history)

else:
    st.info("Enter a review and click **Check Review** to get results.")
