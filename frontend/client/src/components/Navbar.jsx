function Navbar() {
  return (
    <nav className="navbar">
      <div className="logo">🛡️ UPI Fraud Shield</div>

      <ul className="nav-links">
        <li>
          <button
            type="button"
            className="nav-btn"
            onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
          >
            Home
          </button>
        </li>

        <li>
          <button
            type="button"
            className="nav-btn"
            onClick={() =>
              document.getElementById("about")?.scrollIntoView({
                behavior: "smooth",
              })
            }
          >
            About
          </button>
        </li>

        <li>
          <button
            type="button"
            className="nav-btn"
            onClick={() =>
              document.getElementById("how")?.scrollIntoView({
                behavior: "smooth",
              })
            }
          >
            How It Works
          </button>
        </li>

        <li>
          <a
            href="https://github.com/khushists03-arch/UPI-Fraud-Detection-System"
            target="_blank"
            rel="noreferrer"
          >
            GitHub
          </a>
        </li>
      </ul>
    </nav>
  );
}

export default Navbar;