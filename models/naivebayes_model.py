from sklearn.naive_bayes import MultinomialNB

def train_naive_bayes(X_train, y_train):
    model_NB = MultinomialNB()
    model_NB.fit(X_train, y_train)

    return model_NB
