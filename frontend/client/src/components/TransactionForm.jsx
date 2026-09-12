import { useState } from "react";
import ResultCard from "./ResultCard";

function TransactionForm() {
  const [formData, setFormData] = useState({
    amount: "",
    merchant: "",
    transactionType: "Payment",
    location: "",
    description: "",
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(
        "http://127.0.0.1:5000/api/predict",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            amount: Number(formData.amount),
            merchant: formData.merchant.trim(),
            transactionType: formData.transactionType,
            location: formData.location.trim(),
            description: formData.description.trim(),
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.error || "Unable to process transaction."
        );
      }

      if (!data.success) {
        throw new Error(
          data.error || "Prediction failed."
        );
      }

      setResult({
        prediction: data.prediction,
        confidence: Number(data.confidence) * 100,
        fraudScore: Number(data.fraud_score) * 100,
        xgboost: data.xgboost,
        isolationForest: data.isolation_forest,
      });
    } catch (err) {
      console.error("Prediction error:", err);

      setError(
        err.message ||
          "Unable to connect to the fraud detection server."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="form-container">
      <h2>Check a Transaction</h2>

      <form onSubmit={handleSubmit}>
        <input
          type="number"
          name="amount"
          placeholder="Amount (₹)"
          value={formData.amount}
          onChange={handleChange}
          min="0.01"
          step="0.01"
          required
        />

        <input
          type="text"
          name="merchant"
          placeholder="Merchant Name"
          value={formData.merchant}
          onChange={handleChange}
          required
        />

        <select
          name="transactionType"
          value={formData.transactionType}
          onChange={handleChange}
        >
          <option value="Payment">Payment</option>
          <option value="Transfer">Transfer</option>
          <option value="Request Money">
            Request Money
          </option>
        </select>

        <input
          type="text"
          name="location"
          placeholder="Location"
          value={formData.location}
          onChange={handleChange}
          required
        />

        <textarea
          name="description"
          placeholder="Transaction Description"
          value={formData.description}
          onChange={handleChange}
          required
        />

        <button
          type="submit"
          disabled={loading}
        >
          {loading ? "Checking..." : "Predict Fraud"}
        </button>
      </form>

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      <ResultCard result={result} />
    </div>
  );
}

export default TransactionForm;