import { useState } from "react";
import "./App.css";
import { usePrediction } from "./hooks/usePrediction";


function LoanForm() {
  const { predict, loading, error } = usePrediction();
   const [formData, setFormData] = useState({
    Gender: "Male",
    Married: "No",
    Dependents: "0",
    Education: "Graduate",
    Self_Employed: "No",
    ApplicantIncome: "",
    CoapplicantIncome: "",
    LoanAmount: "",
    Loan_Amount_Term: "360",
    Credit_History: "1",
    Property_Area: "Urban",
  });

  const [result, setResult] = useState(null);
  const [showForm, setShowForm] = useState(true);

  const handleChange = (e) => {
    const { name, value } = e.target;

    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = async(e) => {

    e.preventDefault();

    const prediction = await predict(formData);

    setResult(
    prediction === "Approved"
      ? "It will be Approved"
      : "It will be Rejected"
  );

  setShowForm(false);
  };

 

  return (
    <div className="loan-form-container">
      <h1>Loan Approval Prediction</h1>
      {showForm ? (
      <form onSubmit={handleSubmit} className="loan-form">

        <div className="form-group">
          <label>Gender</label>
          <select name="Gender" value={formData.Gender} onChange={handleChange}>
            <option>Male</option>
            <option>Female</option>
          </select>
        </div>

        <div className="form-group">
          <label>Married</label>
          <select name="Married" value={formData.Married} onChange={handleChange}>
            <option>No</option>
            <option>Yes</option>
          </select>
        </div>

        <div className="form-group">
          <label>Dependents</label>
          <select name="Dependents" value={formData.Dependents} onChange={handleChange}>
            <option>0</option>
            <option>1</option>
            <option>2</option>
            <option>3+</option>
          </select>
        </div>

        <div className="form-group">
          <label>Education</label>
          <select name="Education" value={formData.Education} onChange={handleChange}>
            <option>Graduate</option>
            <option>Not Graduate</option>
          </select>
        </div>

        <div className="form-group">
          <label>Self Employed</label>
          <select name="Self_Employed" value={formData.Self_Employed} onChange={handleChange}>
            <option>No</option>
            <option>Yes</option>
          </select>
        </div>

        <div className="form-group">
          <label>Applicant Income</label>
          <input
            type="number"
            name="ApplicantIncome"
            value={formData.ApplicantIncome}
            onChange={handleChange}
            required
          />
        </div>

        <div className="form-group">
          <label>Coapplicant Income</label>
          <input
            type="number"
            name="CoapplicantIncome"
            value={formData.CoapplicantIncome}
            onChange={handleChange}
          />
        </div>

        <div className="form-group">
          <label>Loan Amount</label>
          <input
            type="number"
            name="LoanAmount"
            value={formData.LoanAmount}
            onChange={handleChange}
            required
          />
        </div>

        <div className="form-group">
          <label>Loan Term</label>
          <input
            type="number"
            name="Loan_Amount_Term"
            value={formData.Loan_Amount_Term}
            onChange={handleChange}
          />
        </div>

        <div className="form-group">
          <label>Credit History</label>
          <select
            name="Credit_History"
            value={formData.Credit_History}
            onChange={handleChange}
          >
            <option value="1">1 (Good)</option>
            <option value="0">0 (Bad)</option>
          </select>
        </div>

        <div className="form-group">
          <label>Property Area</label>
          <select
            name="Property_Area"
            value={formData.Property_Area}
            onChange={handleChange}
          >
            <option>Urban</option>
            <option>Semiurban</option>
            <option>Rural</option>
          </select>
        </div>

        <button className="predict-button">Predict</button>

      </form>

      ):(<h1 style={{color:`${(result=="It will be Rejected")?"red":"green"}`}}>{result}</h1>)}
      {loading && <p>Loading...</p>}
      {error && <p>Error: {error}</p>}
    </div>
  );
}

export default function App() {
  return <LoanForm />;
}