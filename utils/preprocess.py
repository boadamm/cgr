import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

def preprocess_and_vectorize(df, text_column="text_", max_features=5000):
    df["cleaned_text"] = df[text_column].astype(str).apply(clean_text)
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(df["cleaned_text"])
    return X, df["label"].map({"CG": 1, "OR": 0}), vectorizer
