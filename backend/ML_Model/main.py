from extraction import preprocess_data,missingField,check_low_variance
from model import model_V1, predict, accuracy,model_v2,model_V3,model_v4,threshold_search,model_v5,predict_v2,model_v6,model_v7,forest_predict
if __name__ == "__main__":

    X_train, X_test, y_train, y_test = preprocess_data("datasets/train.csv")
    # df=missingField("datasets/train.csv")
    # check_low_variance(df,"LoanAmount")
    print("Training Model Start")
    # ==== model v2 code=====
    # W, b = model_v2(X_train, y_train, nf=X_train.shape[1])
    # print("Training Completed")
    # y_pred = predict(X_test, W, b)
    # acc = accuracy(y_test, y_pred)
    # print("Final Accuracy:", acc)
    
    
    #==== model v3 code=====
    # model_V3(X_train, y_train,X_test,y_test)
    # print("Training Completed")
    
    #==== model v4 code=====
    # W, b = model_v4(X_train, y_train)
    # print("Training Completed")
    # print("Threshold Search Start")
    # threshold = threshold_search(X_test, y_test, W, b)
    # y_pred = predict(X_test, W, b, threshold)
    # acc = accuracy(y_test, y_pred)
    # print("Final Accuracy:", acc)
    
    #==== model v5 code=====
    # tree=model_v5(X_train, y_train)
    # p=predict_v2(X_test, tree)
    # acc = accuracy(y_test, p)
    # print("Final Accuracy:", acc)
    # print("Training Completed")
    
    #==== model v6 code=====
    # model_v6(X_train, y_train,X_test,y_test)
    # print("Training Completed")
    #==== model v7 code=====
    forest=model_v7(X_train, y_train)
    p=forest_predict(X_test, forest)
    acc = accuracy(y_test, p)
    print("Final Accuracy:", acc)
    print("Training Completed")