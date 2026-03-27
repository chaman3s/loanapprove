import { useState } from "react";

export function usePrediction() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const predict = async (formData) => {
    setLoading(true);
    setError(null);

    try {
      const res = await fetch("https://loanapprove.onrender.com/api/v1/users/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      const data = await res.json();

      setLoading(false);
      return data.prediction;

    } catch (err) {
      setLoading(false);
      setError("Something went wrong");
      console.log(err);
      return null;
    }
  };

  return { predict, loading, error };
}