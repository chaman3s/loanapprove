import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def preprocess_data(dataset_path):

    df = pd.read_csv(dataset_path)

    df = df.drop("Loan_ID", axis=1)

    df["Dependents"] = df["Dependents"].replace("3+", 3)

    df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
    df["Married"] = df["Married"].fillna(df["Married"].mode()[0])
    df["Dependents"] = df["Dependents"].fillna(df["Dependents"].mode()[0])
    df["Self_Employed"] = df["Self_Employed"].fillna(df["Self_Employed"].mode()[0])

    df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].mean())
    df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0])
    df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].mode()[0])

    df["Gender"] = df["Gender"].map({"Male":1,"Female":0})
    df["Married"] = df["Married"].map({"Yes":1,"No":0})
    df["Education"] = df["Education"].map({"Graduate":1,"Not Graduate":0})
    df["Self_Employed"] = df["Self_Employed"].map({"Yes":1,"No":0})
    df["Loan_Status"] = df["Loan_Status"].map({"Y":1,"N":0})

    df["Property_Area"] = df["Property_Area"].map({
        "Rural":0,
        "Semiurban":1,
        "Urban":2
    })

    df["Dependents"] = df["Dependents"].astype(int)
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["LoanIncomeRatio"] = df["LoanAmount"] / df["TotalIncome"]
    df = df.drop(["ApplicantIncome","CoapplicantIncome"], axis=1)
    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = X_train.values
    X_test = X_test.values

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    y_train = y_train.values
    y_test = y_test.values

    print("Train Shape:", X_train.shape)
    print("Test Shape:", X_test.shape)

    return X_train, X_test, y_train, y_test