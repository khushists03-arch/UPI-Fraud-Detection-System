import "./App.css";
import Navbar from "./components/Navbar";
import Hero from "./components/Hero";
import FeatureCards from "./components/FeatureCards";
import TransactionForm from "./components/TransactionForm";
import Footer from "./components/Footer";

function App() {
  return (
    <>
      <Navbar />
      <Hero />
      <FeatureCards />
      <TransactionForm />
      <Footer />
    </>
  );
}

export default App;