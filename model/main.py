from model import LogisticModel

def main():
    logistic_model = LogisticModel('data.csv')
    
    logistic_model.preprocess()

    X_train, X_test, Y_train, Y_test = logistic_model.split_data()

    X_train_scaled, X_test_scaled = logistic_model.scale_features(X_train, X_test)

    logistic_model.train_model(X_train_scaled, Y_train)

    cm, accuracy = logistic_model.evaluate_model(X_test_scaled, Y_test)
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {accuracy:.4f}")
    
    logistic_model.handle_categorical()

    X_train, X_test, Y_train, Y_test = logistic_model.split_data(test_size=0.55)
    X_train_scaled, X_test_scaled = logistic_model.scale_features(X_train, X_test)

    logistic_model.train_model(X_train_scaled, Y_train)

    cm, new_accuracy = logistic_model.evaluate_model(X_test_scaled, Y_test)
    print(f"New Confusion Matrix:\n{cm}")
    print(f"New Accuracy: {new_accuracy:.4f}")
    
    cross_val_scores = logistic_model.cross_validate(cv_folds=5)
    print(f"Cross-validation scores: {cross_val_scores}")
    print(f"Mean Cross-validation score: {cross_val_scores.mean():.4f}")

if __name__ == "__main__":
    main()
