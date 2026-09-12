function ResultCard({ result }) {
  if (!result) {
    return null;
  }

  const isFraud = result.prediction === "Fraud";

  return (
    <div
      className={`result-card ${
        isFraud ? "fraud" : "safe"
      }`}
    >
      <h2>
        {isFraud
          ? "🔴 Fraud Detected"
          : "🟢 Safe Transaction"}
      </h2>

      <p>
        Confidence:{" "}
        {Number(result.confidence).toFixed(1)}%
      </p>

      {result.fraudScore !== undefined && (
        <p>
          Fraud Score:{" "}
          {Number(result.fraudScore).toFixed(1)}%
        </p>
      )}
    </div>
  );
}

export default ResultCard;