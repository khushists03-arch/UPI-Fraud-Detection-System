function FeatureCards() {
  const features = [
    {
      icon: "⚡",
      title: "Instant Detection",
      desc: "Real-time fraud prediction within seconds.",
    },
    {
      icon: "🤖",
      title: "AI Powered",
      desc: "Machine Learning models detect suspicious transactions.",
    },
    {
      icon: "🔒",
      title: "Secure Payments",
      desc: "Protect users from fraudulent UPI activities.",
    },
  ];

  return (
    <div className="feature-grid">
      {features.map((feature, index) => (
        <div key={index} className="feature-card">
          <div className="feature-icon">{feature.icon}</div>
          <h3>{feature.title}</h3>
          <p>{feature.desc}</p>
        </div>
      ))}
    </div>
  );
}

export default FeatureCards;