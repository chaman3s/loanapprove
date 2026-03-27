import numpy as np
def transform_input(data):
    gender = 1 if data.Gender == "Male" else 0
    married = 1 if data.Married == "Yes" else 0
    education = 1 if data.Education == "Graduate" else 0
    self_emp = 1 if data.Self_Employed == "Yes" else 0
    property_area = {
        "Rural": 0,
        "Semiurban": 1,
        "Urban": 2
    }[data.Property_Area]
    dependents = 3 if data.Dependents == "3+" else int(data.Dependents)
    applicant_income = float(data.ApplicantIncome)
    co_income = float(data.CoapplicantIncome)
    loan_amount = float(data.LoanAmount)
    loan_term = float(data.Loan_Amount_Term)
    credit_history = float(data.Credit_History)
    total_income = applicant_income + co_income
    loan_income_ratio = loan_amount / (total_income + 1)
    IncomePerDependent= total_income / (dependents + 1)
    features = np.array([[
        gender,
        married,
        dependents,
        self_emp,
        education,
        loan_amount,
        loan_term,
        credit_history,
        property_area,
        total_income,
        loan_income_ratio,
        IncomePerDependent
    ]],dtype=float)
    # Add squared features to match the training data preprocessing
    return np.concatenate([features, features**2], axis=1)