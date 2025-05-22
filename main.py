import pandas as pd
import time
import joblib

from sklearn.model_selection import train_test_split

from models.distilbert_model import train_distilbert
from models.lstm_model import train_lstm
from models.logreg_model import train_logreg
from models.ranfor_model import train_random_forest
from models.naivebayes_model import train_naive_bayes

from utils.preprocess import preprocess_and_vectorize
from utils.plotting import plot_confusion_matrix
from utils.evaluate import model_report

# Load dataset
df = pd.read_csv("data/fake_reviews_act.csv")

# Preprocess and extract features
X, y, tfidf_vectorizer = preprocess_and_vectorize(df, text_column="text_")

joblib.dump(tfidf_vectorizer, "outputs/tfidf_vectorizer.pkl")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
start_time= time.time()
model_LR = train_logreg(X_train, y_train)
end_time = time.time()
print("\nLogistic Regression Evaluation:")
y_pred_LR = model_report(model_LR, X_test, y_test, start_time, end_time)


# Random Forest
start_time= time.time()
model_RF = train_random_forest(X_train, y_train)
end_time = time.time()
print("\nRandom Forest Evaluation:")
y_pred_RF = model_report(model_RF, X_test, y_test, start_time, end_time)


# Naive Bayes
start_time= time.time()
model_NB = train_naive_bayes(X_train, y_train)
end_time = time.time()
print("\nNaive Bayes Evaluation:")
y_pred_NB = model_report(model_NB, X_test, y_test, start_time, end_time)

# LSTM
model_LSTM = train_lstm(df)


# Distilbert
start_time= time.time()
model_BERT = train_distilbert(df)
end_time = time.time()
print(f"Runtime BERT = {start_time - end_time}")


# For sklearn models
joblib.dump(model_LR, "outputs/logreg.pkl")
joblib.dump(model_RF, "outputs/ranfor.pkl")
joblib.dump(model_NB, "outputs/naivebayes.pkl")

# For LSTM
model_LSTM.save("outputs/lstm_model.h5")

# Plot confusion matrix for models
print("\nConfusion Matrix for Logistic Regression:")
plot_confusion_matrix(y_test, y_pred_LR)

print("\nConfusion Matrix for Random Forest:")
plot_confusion_matrix(y_test, y_pred_RF)

print("\nConfusion Matrix for Naive Bayes:")
plot_confusion_matrix(y_test, y_pred_NB)

