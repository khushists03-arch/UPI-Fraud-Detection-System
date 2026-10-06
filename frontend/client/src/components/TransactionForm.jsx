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

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    const mockResult = {
      prediction: Math.random() > 0.5 ? "Fraud" : "Safe",
      confidence: Math.floor(Math.random() * 15) + 85,
    };

    setResult(mockResult);
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
          <option>Payment</option>
          <option>Transfer</option>
          <option>Request Money</option>
        </select>

        <input
          type="text"
          name="location"
          placeholder="Location"
          value={formData.location}
          onChange={handleChange}
        />

        <textarea
          name="description"
          placeholder="Transaction Description"
          value={formData.description}
          onChange={handleChange}
        />

        <button type="submit">Predict Fraud</button>
      </form>

      <ResultCard result={result} />
    </div>
  );
}

export default TransactionForm;