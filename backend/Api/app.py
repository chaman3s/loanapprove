from fastapi import FastAPI
from pydantic import BaseModel
from supabase import create_client
import joblib
from ML_Model.model import forest_predict
import numpy as np
from dotenv import load_dotenv
import os
from utils import transform_input
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
url = os.getenv("SUPABASE_URL")
apiKey=os.getenv("SUPABASE_KEY")
print("URL:", url)
print("KEY:", apiKey)
supabase = create_client(url, apiKey)
model = joblib.load("ML_Model/model.pkl")
app=FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class LoanForm(BaseModel):
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    ApplicantIncome: str
    CoapplicantIncome: str
    LoanAmount: str
    Loan_Amount_Term: str
    Credit_History: str
    Property_Area: str

@app.post("/api/v1/users/predict")
def predict(data: LoanForm):
    x=transform_input(data)
    predict = forest_predict(x, model)
    result = "Approved" if predict[0] == 1 else "Rejected" 
    dt ={
        "gender": data.Gender,
        "married": data.Married,
        "dependents": data.Dependents,
        "education": data.Education,
        "self_Employed": data.Self_Employed,
        "applicantIncome": data.ApplicantIncome,
        "coapplicantIncome": data.CoapplicantIncome,
        "loanAmount": data.LoanAmount,
        "loan_Amount_Term": data.Loan_Amount_Term,
        "credit_History": data.Credit_History,
        "property_Area": data.Property_Area,
        "result": result
    }
    try:
        supabase.table("loans").delete().execute()
    except Exception as e:
        print(e)
    supabase.table("loans").insert(dt).execute()
    return {
        "prediction": result
    }