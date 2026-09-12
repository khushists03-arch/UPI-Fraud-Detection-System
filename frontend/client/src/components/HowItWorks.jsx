function HowItWorks() {
  return (
    <section className="how-it-works" id="how">
      <h2>How It Works</h2>

      <div className="steps">
        <div className="step">
          <h3>1. Enter Transaction</h3>
          <p>Fill in the transaction details.</p>
        </div>

        <div className="step">
          <h3>2. AI Analysis</h3>
          <p>XGBoost and Isolation Forest analyze fraud patterns.</p>
        </div>

        <div className="step">
          <h3>3. Get Result</h3>
          <p>Receive Safe or Fraud prediction with confidence.</p>
        </div>
      </div>
    </section>
  );
}

export default HowItWorks;