from extraction import preprocess_data
from model import model, predict, accuracy

if __name__ == "__main__":

    X_train, X_test, y_train, y_test = preprocess_data("datasets/train.csv")

    print("Training Model Start")

    W, b = model(X_train, y_train, nf=X_train.shape[1])

    print("Training Completed")

    y_pred = predict(X_test, W, b)

    acc = accuracy(y_test, y_pred)

    print("Final Accuracy:", acc)