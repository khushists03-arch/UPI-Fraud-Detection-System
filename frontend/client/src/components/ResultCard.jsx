function ResultCard({ result }) {
  if (!result) {
    return null;
  }


  const isFraud = result.prediction === "Fraud";


  /*
   * TransactionForm has already converted these values
   * from decimals into percentages.
   *
   * Example:
   *
   * confidence = 99.9961
   * fraudScore = 0.0039
   *
   * Therefore, DO NOT multiply by 100 here.
   */

  const confidence = Number(result.confidence);

  const fraudScore = Number(result.fraudScore);


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
        Confidence: {confidence.toFixed(4)}%
      </p>


      <p>
        Fraud Score: {fraudScore.toFixed(4)}%
      </p>

    </div>
  );
}


export default ResultCard;