function ResultCard({ result }) {
  if (!result) return null;

  return (
    <div className={`result-card ${result.prediction === "Fraud" ? "fraud" : "safe"}`}>
      <h2>{result.prediction === "Fraud" ? "🔴 Fraud Detected" : "🟢 Safe Transaction"}</h2>

      <p>Confidence: {result.confidence}%</p>

      <div className="confidence-bar">
        <div
          className="confidence-fill"
          style={{ width: `${result.confidence}%` }}
        ></div>
      </div>

      <p>
        Risk Level:{" "}
        {result.prediction === "Fraud" ? "High" : "Low"}
      </p>
    </div>
  );
}

export default ResultCard;