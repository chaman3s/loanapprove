from extraction import preprocess_data
from model import model_v7,forest_predict,accuracy
import joblib
if __name__ == "__main__":

    X_train, X_test, y_train, y_test = preprocess_data("datasets/train.csv")

    print("Training Model Start")
   
    forest=model_v7(X_train, y_train)
    joblib.dump(forest, "model.pkl")
    p=forest_predict(X_test, forest)
    acc = accuracy(y_test, p)
    print("Final Accuracy:", acc)
    print("Training Completed")