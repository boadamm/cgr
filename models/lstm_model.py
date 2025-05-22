import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from utils.preprocess import clean_text
from utils.plotting import plot_confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def preprocess_for_lstm(df, text_column="text_", num_words=10000, max_length=100):
    df["cleaned_text"] = df[text_column].astype(str).apply(clean_text)
    texts = df["cleaned_text"].values
    labels = df["label"].map({"CG": 1, "OR": 0}).values

    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)

    joblib.dump(tokenizer, "outputs/lstm_tokenizer.pkl")

    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

    return padded_sequences, labels, tokenizer

def build_lstm_model(vocab_size=10000, embedding_dim=64, max_length=100):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        LSTM(64),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_lstm(df):
    X, y, tokenizer = preprocess_for_lstm(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_lstm_model()

    start_time = time.time()
    model.fit(X_train, y_train, epochs=5, batch_size=16, validation_data=(X_test, y_test), callbacks=[EarlyStopping(patience=3)])
    end_time = time.time()

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    # Evaluation
    print("\nLSTM Evaluation:")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"\nRuntime: {end_time - start_time:.4f} seconds\n")
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=["Original", "Fake"], zero_division=0))

    plot_confusion_matrix(y_test, y_pred, labels=["Original", "Fake"], title="Confusion Matrix - LSTM")

    return model
