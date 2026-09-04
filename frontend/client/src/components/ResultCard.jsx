function ResultCard({ result }) {
  if (!result) return null;

  return (
    <div className={`result-card ${result.prediction === "Fraud" ? "fraud" : "safe"}`}>
      <h2>
        {result.prediction === "Fraud"
          ? "🔴 Fraud Detected"
          : "🟢 Safe Transaction"}
      </h2>

      <p>Confidence: {result.confidence}%</p>
    </div>
  );
}

export default ResultCard;