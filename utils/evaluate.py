from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def model_report(model, X_test, y_test, start_time, end_time):
    y_pred = model.predict(X_test)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"\nRuntime: {end_time - start_time:.4f} seconds")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Original", "Fake"]))

    return y_pred