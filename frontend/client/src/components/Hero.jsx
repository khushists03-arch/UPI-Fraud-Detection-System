function Hero() {
  return (
    <section className="hero" id="home">
      <div className="hero-text">
        <h1>Detect UPI Fraud Instantly</h1>

        <p>
          AI-powered fraud detection using Machine Learning,
          XGBoost and Isolation Forest.
        </p>

        <button
          className="hero-btn"
          onClick={() =>
            document
              .querySelector(".form-container")
              ?.scrollIntoView({ behavior: "smooth" })
          }
        >
          Check a Transaction
        </button>
      </div>
    </section>
  );
}

export default Hero;