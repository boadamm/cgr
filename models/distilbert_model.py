import tensorflow as tf
import keras

from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from utils.preprocess import clean_text

if not hasattr(keras.utils, 'unpack_x_y_sample_weight'):
    keras.utils.unpack_x_y_sample_weight = tf.keras.utils.unpack_x_y_sample_weight

def prepare_data_for_distilbert(df, text_column="text_", label_column="label"):
    df["cleaned_text"] = df[text_column].astype(str).apply(clean_text)
    texts = df["cleaned_text"].tolist()
    labels = df[label_column].map({"CG": 1, "OR": 0}).tolist()
    return texts, labels

def train_distilbert(df, epochs=1, batch_size=16, max_length=128):

    texts, labels = prepare_data_for_distilbert(df)
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=max_length)
    test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=max_length)

    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(train_encodings),
        y_train
    )).batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((
        dict(test_encodings),
        y_test
    )).batch(batch_size)

    model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = ['accuracy']

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    history = model.fit(train_dataset, validation_data=test_dataset, epochs=epochs)
    
    pred_logits = model.predict(test_dataset).logits
    y_pred = tf.argmax(pred_logits, axis=1).numpy()

    print("\nDistilBERT Evaluation:")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Save Model
    model.save_pretrained("outputs/bert_model/")
    tokenizer.save_pretrained("outputs/bert_model/")

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    labels = ['Original', 'Fake']
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - DistilBERT')
    plt.tight_layout()
    plt.show()

    return model, tokenizer, history
