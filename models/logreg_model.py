from sklearn.linear_model import LogisticRegression

def train_logreg(X_train, y_train):
    model_LR = LogisticRegression()
    model_LR.fit(X_train, y_train)

    return model_LR