import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { createPortal } from "react-dom";
import App from "./App.jsx";
import "./styles.css";
import "./theme.css";

function ThemeToggle() {
  const [theme, setTheme] = useState(() => {
    try {
      return localStorage.getItem("signal-theme") === "dark" ? "dark" : "light";
    } catch {
      return "light";
    }
  });
  const [header, setHeader] = useState(null);

  useEffect(() => {
    document.body.dataset.theme = theme;
    try {
      localStorage.setItem("signal-theme", theme);
    } catch {
      // The selected theme still works if browser storage is unavailable.
    }
  }, [theme]);

  useEffect(() => {
    setHeader(document.querySelector(".topbar"));
  }, []);

  if (!header) return null;

  return createPortal(
    <button
      className="theme-toggle"
      type="button"
      aria-label={`Switch to ${theme === "light" ? "night" : "light"} mode`}
      aria-pressed={theme === "dark"}
      onClick={() => setTheme((current) => current === "light" ? "dark" : "light")}
    >
      <svg aria-hidden="true" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        {theme === "light" ? (
          <><path d="M20.5 15.5A8.5 8.5 0 0 1 8.5 3.5 8.5 8.5 0 1 0 20.5 15.5Z" /></>
        ) : (
          <><circle cx="12" cy="12" r="4" /><path d="M12 2v2m0 16v2M4.93 4.93l1.42 1.42m11.3 11.3 1.42 1.42M2 12h2m16 0h2M4.93 19.07l1.42-1.42m11.3-11.3 1.42-1.42" /></>
        )}
      </svg>
      <span>{theme === "light" ? "Night mode" : "Light mode"}</span>
    </button>,
    header,
  );
}

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <>
      <App />
      <ThemeToggle />
    </>
  </React.StrictMode>,
);
