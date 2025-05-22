# Fake Review Detection System

This project is a multi-model system designed to detect fake product reviews. It utilizes a combination of traditional machine learning models and deep learning models to classify reviews as either genuine (Original) or fake (Computer Generated).

## Features

*   Analyzes product reviews to predict authenticity.
*   Employs five different machine learning models:
    *   Logistic Regression
    *   Random Forest
    *   Naive Bayes
    *   LSTM (Long Short-Term Memory)
    *   DistilBERT
*   Provides individual model predictions and a consensus decision based on majority voting.
*   Web interface built with Streamlit for easy interaction.

## Project Structure

```
├── app.py              # Main Streamlit application for inference
├── main.py             # Script for training models
├── requirements.txt    # Project dependencies
├── data/               # Directory for datasets
│   └── fake_reviews_act.csv # Example dataset
├── models/             # Python modules for model definitions and training functions
│   ├── distilbert_model.py
│   ├── logreg_model.py
│   ├── lstm_model.py
│   ├── naivebayes_model.py
│   └── ranfor_model.py
├── utils/              # Utility scripts (preprocessing, evaluation, plotting)
│   ├── preprocess.py
│   ├── evaluate.py
│   └── plotting.py
├── outputs/            # Directory for saved models, tokenizers, and vectorizers (ignored by Git)
├── .gitignore          # Specifies intentionally untracked files that Git should ignore
└── README.md           # This file
```

## Setup and Installation

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone https://github.com/boadamm/cgr.git
    cd cgr
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK resources (if running for the first time):**
    The `utils/preprocess.py` script attempts to download `stopwords` and `wordnet` from NLTK. This might require running Python once or manually triggering the downloads:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    ```

## Usage

### 1. Training the Models

To train all the models and save them to the `outputs/` directory:
```bash
python main.py
```
This script will:
* Load the dataset from `data/fake_reviews_act.csv`.
* Preprocess the data.
* Train Logistic Regression, Random Forest, Naive Bayes, LSTM, and DistilBERT models.
* Save the trained models, TF-IDF vectorizer, and LSTM tokenizer.
* Print evaluation reports and display confusion matrices for the models.

### 2. Running the Application

To start the Streamlit web application:
```bash
streamlit run app.py
```
Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`). You can then enter a review into the text area and click "Check Review" to see the predictions from all models.

## Models Used

*   **Logistic Regression**
*   **Random Forest**
*   **Naive Bayes**
*   **LSTM (Long Short-Term Memory)**
*   **DistilBERT**

The application displays predictions from each model and a final consensus based on a majority vote.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[Specify your license here, e.g., MIT License] 