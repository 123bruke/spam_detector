import { useEffect, useState } from "react";

const samples = [
  {
    label: "Suspicious offer",
    text: "Congratulations! You have won a free prize. Call now to claim your reward.",
  },
  {
    label: "Everyday message",
    text: "Hey, are we still meeting at six near the library?",
  },
];

const modelOptions = [
  { id: "naive_bayes", name: "Naive Bayes", detail: "Explainable · token evidence" },
  { id: "neural_network", name: "Neural network", detail: "PyTorch · pattern learning" },
];

const metricLabels = [
  ["accuracy", "Accuracy"],
  ["precision", "Precision"],
  ["recall", "Recall"],
  ["f1", "F1 score"],
  ["roc_auc", "ROC-AUC"],
];

function Icon({ name, size = 18 }) {
  const shared = {
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.8,
    strokeLinecap: "round",
    strokeLinejoin: "round",
  };
  const shapes = {
    shield: <><path d="M12 3 19 6v5c0 4.4-2.8 7.8-7 10-4.2-2.2-7-5.6-7-10V6l7-3Z" /><path d="m9 12 2 2 4-4" /></>,
    spark: <><path d="m12 3 1.7 5.3L19 10l-5.3 1.7L12 17l-1.7-5.3L5 10l5.3-1.7L12 3Z" /><path d="m19 15 .8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8L19 15Z" /></>,
    arrow: <><path d="M5 12h14" /><path d="m13 6 6 6-6 6" /></>,
    check: <path d="m5 12 4 4L19 6" />,
    alert: <><path d="M10.3 4.3 2.8 17.2A2 2 0 0 0 4.5 20h15a2 2 0 0 0 1.7-2.8L13.7 4.3a2 2 0 0 0-3.4 0Z" /><path d="M12 9v4m0 3h.01" /></>,
    rotate: <><path d="M20 7v5h-5" /><path d="M4.9 16A8 8 0 0 0 18.4 18L20 12M4 17v-5h5" /><path d="M5.6 6A8 8 0 0 1 19.1 8L20 12" /></>,
    lock: <><rect x="4" y="10" width="16" height="11" rx="2" /><path d="M8 10V7a4 4 0 0 1 8 0v3" /><path d="M12 14v3" /></>,
    activity: <><path d="M3 12h4l3-8 4 16 3-8h4" /></>,
    message: <><path d="M21 11.5a8.4 8.4 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.4 8.4 0 0 1-3.8-.9L3 21l1.9-5.7a8.4 8.4 0 0 1-.9-3.8A8.5 8.5 0 0 1 8.7 3.9a8.4 8.4 0 0 1 3.8-.9h.5a8.5 8.5 0 0 1 8 8v.5Z" /></>,
    close: <><path d="m18 6-12 12M6 6l12 12" /></>,
    info: <><circle cx="12" cy="12" r="9" /><path d="M12 11v5m0-8h.01" /></>,
  };

  return (
    <svg
      aria-hidden="true"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      {...shared}
    >
      {shapes[name]}
    </svg>
  );
}

function formatPercent(value) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "—";
}

function App() {
  const [message, setMessage] = useState("");
  const [model, setModel] = useState("naive_bayes");
  const [threshold, setThreshold] = useState(0.5);
  const [result, setResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState("");
  const [health, setHealth] = useState("checking");
  const [metrics, setMetrics] = useState(null);
  const [metricsError, setMetricsError] = useState("");

  useEffect(() => {
    let active = true;
    async function loadDashboard() {
      try {
        const response = await fetch("/api/health");
        if (!response.ok) throw new Error("Could not reach the prediction API.");
        const data = await response.json();
        if (active) setHealth(data.models_ready ? "ready" : "training");
      } catch {
        if (active) setHealth("offline");
      }

      try {
        const response = await fetch("/api/metrics");
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || "Evaluation metrics are not available.");
        }
        if (active) setMetrics(data);
      } catch (loadError) {
        if (active) setMetricsError(loadError.message);
      }
    }
    loadDashboard();
    return () => {
      active = false;
    };
  }, []);

  async function analyze(event) {
    event.preventDefault();
    if (!message.trim()) {
      setError("Add a message first, then run the scan.");
      return;
    }
    setError("");
    setIsAnalyzing(true);
    setResult(null);
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, model, threshold }),
      });
      const data = await response.json();
      if (!response.ok) {
        const detail = typeof data.detail === "string" ? data.detail : "The scan could not be completed.";
        throw new Error(detail);
      }
      setResult(data);
    } catch (scanError) {
      setError(scanError.message || "The scan could not be completed.");
    } finally {
      setIsAnalyzing(false);
    }
  }

  function chooseSample(text) {
    setMessage(text);
    setResult(null);
    setError("");
  }

  const modelMetrics = metrics?.test_metrics || {};
  const bayesMetrics = modelMetrics["Multinomial Naive Bayes"];
  const neuralMetrics = modelMetrics["Neural Network (PyTorch MLP)"];
  const scorePercent = result ? Math.round(result.spam_score * 100) : 0;
  const statusText = {
    checking: "Connecting to models",
    ready: "Models ready",
    training: "Training required",
    offline: "API offline",
  }[health];

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#" aria-label="Signal home">
          <span className="brand-mark"><Icon name="shield" size={22} /></span>
          <span className="brand-copy">
            <strong>signal<span>.</span></strong>
            <small>MESSAGE SECURITY LAB</small>
          </span>
        </a>
        <nav className="main-nav" aria-label="Main navigation">
          <a className="active" href="#scanner">Scanner</a>
          <a href="#performance">Model performance</a>
        </nav>
        <div className={`service-status status-${health}`} role="status">
          <span className="status-dot" />
          {statusText}
        </div>
      </header>

      <main>
        <section className="hero">
          <div className="hero-content">
            <div className="eyebrow"><span className="eyebrow-line" /> YOUR SMARTER MESSAGE FILTER</div>
            <h1>Read between<br />the <span>messages.</span></h1>
            <p>Two machine-learning models. One clearer view of what lands in your inbox.</p>
            <div className="hero-points">
              <span><Icon name="activity" size={15} /> Real-time analysis</span>
              <span><Icon name="lock" size={15} /> Runs on your machine</span>
            </div>
          </div>
          <div className="hero-art" aria-hidden="true">
            <div className="orbit orbit-one" />
            <div className="orbit orbit-two" />
            <div className="orbit-core"><Icon name="shield" size={54} /></div>
            <span className="orbit-node node-one"><Icon name="message" size={17} /></span>
            <span className="orbit-node node-two"><Icon name="spark" size={18} /></span>
            <span className="orbit-node node-three"><Icon name="check" size={17} /></span>
          </div>
        </section>

        <section className="workspace section-wrap" id="scanner">
          <div className="section-heading">
            <div>
              <div className="eyebrow"><span className="eyebrow-line" /> MESSAGE SCANNER</div>
              <h2>Put a message to the test</h2>
            </div>
            <span className="section-index">01 <span>/ 02</span></span>
          </div>

          <div className="dashboard-grid">
            <form className="panel scanner-panel" onSubmit={analyze}>
              <div className="panel-heading">
                <div className="panel-icon"><Icon name="message" /></div>
                <div><h3>Analyze a message</h3><p>Choose a model and paste in the text.</p></div>
              </div>

              <div className="field-label-row">
                <label className="field-label" htmlFor="message">MESSAGE CONTENT</label>
                <span className="character-count">{message.length.toLocaleString()} / 5,000</span>
              </div>
              <textarea
                id="message"
                maxLength={5000}
                value={message}
                onChange={(event) => {
                  setMessage(event.target.value);
                  setError("");
                  setResult(null);
                }}
                placeholder="Paste an SMS or short message here..."
                rows={5}
              />
              <div className="sample-row">
                <span>Try a sample</span>
                {samples.map((sample) => (
                  <button className="sample-chip" type="button" key={sample.label} onClick={() => chooseSample(sample.text)}>
                    {sample.label}
                  </button>
                ))}
              </div>

              <div className="field-label model-field-label">DETECTION MODEL</div>
              <div className="model-selector" role="group" aria-label="Detection model">
                {modelOptions.map((option) => (
                  <button
                    className={`model-option ${model === option.id ? "selected" : ""}`}
                    type="button"
                    key={option.id}
                    aria-pressed={model === option.id}
                    onClick={() => {
                      setModel(option.id);
                      setResult(null);
                    }}
                  >
                    <span className="model-radio"><span /></span>
                    <span className="model-option-copy"><strong>{option.name}</strong><small>{option.detail}</small></span>
                    {model === option.id && <Icon name="check" size={17} />}
                  </button>
                ))}
              </div>

              <div className="threshold-heading">
                <label className="field-label" htmlFor="threshold">SPAM THRESHOLD</label>
                <output htmlFor="threshold">{Math.round(threshold * 100)}%</output>
              </div>
              <input
                id="threshold"
                className="threshold-slider"
                type="range"
                min="0.1"
                max="0.9"
                step="0.05"
                value={threshold}
                style={{ "--range-progress": `${((threshold - 0.1) / 0.8) * 100}%` }}
                onChange={(event) => {
                  setThreshold(Number(event.target.value));
                  setResult(null);
                }}
              />
              <div className="range-labels"><span>More sensitive</span><span>Fewer false alarms</span></div>

              {error && <div className="inline-error" role="alert"><Icon name="alert" size={17} />{error}</div>}
              <button className="analyze-button" type="submit" disabled={isAnalyzing || health === "offline"}>
                <span>{isAnalyzing ? "Analyzing message…" : "Analyze message"}</span>
                {isAnalyzing ? <span className="button-spinner" /> : <Icon name="arrow" size={19} />}
              </button>
              <div className="privacy-note"><Icon name="lock" size={14} /> Your message is analyzed locally and is not stored.</div>
            </form>

            <section className={`panel result-panel ${result ? "has-result" : ""}`} aria-live="polite">
              <div className="panel-heading result-heading">
                <div className="panel-icon"><Icon name="spark" /></div>
                <div><h3>Scan results</h3><p>{result ? result.model_name : "Your analysis will appear here."}</p></div>
                {result && <button className="icon-button" type="button" aria-label="Clear results" onClick={() => setResult(null)}><Icon name="close" /></button>}
              </div>

              {result ? (
                <div className="result-content">
                  <div className={`verdict ${result.is_spam ? "verdict-spam" : "verdict-ham"}`}>
                    <span className="verdict-icon"><Icon name={result.is_spam ? "alert" : "check"} size={20} /></span>
                    <span><small>{result.is_spam ? "SUSPICIOUS MESSAGE" : "MESSAGE LOOKS CLEAR"}</small><strong>{result.is_spam ? "Likely spam" : "Likely legitimate"}</strong></span>
                  </div>
                  <div className="score-layout">
                    <div className="score-ring" style={{ "--score": `${scorePercent}%` }}>
                      <div><strong>{scorePercent}<small>%</small></strong><span>SPAM SCORE</span></div>
                    </div>
                    <div className="score-explainer">
                      <div className="score-track"><span style={{ width: `${scorePercent}%` }} /></div>
                      <div className="score-scale"><span>0% · Ham</span><span>Spam · 100%</span></div>
                      <div className="threshold-marker"><span>Decision threshold</span><strong>{Math.round(result.threshold * 100)}%</strong></div>
                    </div>
                  </div>
                  <div className="result-note"><Icon name="info" size={16} /><span>{result.score_note}</span></div>
                  {result.evidence_tokens?.length > 0 && (
                    <div className="evidence-section">
                      <div className="evidence-title"><strong>What influenced the score</strong><span>TOP TOKEN SIGNALS</span></div>
                      <div className="token-list">
                        {result.evidence_tokens.slice(0, 6).map((token) => (
                          <span className={`token ${token.log_odds_contribution >= 0 ? "token-spam" : "token-ham"}`} key={token.token}>
                            {token.token}<small>{token.log_odds_contribution >= 0 ? "spam" : "ham"}</small>
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  {result.evidence_tokens?.length === 0 && (
                    <div className="empty-evidence">No message tokens matched the learned vocabulary, so the model relied mostly on its prior.</div>
                  )}
                </div>
              ) : (
                <div className="result-empty">
                  <div className="empty-orbit"><div><Icon name="shield" size={38} /></div></div>
                  <strong>Ready when you are</strong>
                  <p>Run a scan to see the spam score, verdict, and the signals behind the decision.</p>
                  <div className="empty-legend"><span><i className="legend-dot spam-dot" /> Spam signal</span><span><i className="legend-dot ham-dot" /> Legitimate signal</span></div>
                </div>
              )}
            </section>
          </div>
        </section>

        <section className="performance section-wrap" id="performance">
          <div className="section-heading">
            <div>
              <div className="eyebrow"><span className="eyebrow-line" /> THE NUMBERS</div>
              <h2>Models, measured</h2>
            </div>
            <span className="section-index">02 <span>/ 02</span></span>
          </div>

          {metricsError ? (
            <div className="metrics-unavailable"><Icon name="info" size={18} />{metricsError}</div>
          ) : (
            <>
              <div className="metrics-context">
                <span><Icon name="activity" size={16} /> Held-out test set</span>
                {metrics && <span>{metrics.dataset_rows_after_deduplication?.toLocaleString()} messages · {metrics.dataset_source}</span>}
              </div>
              <div className="comparison-grid">
                {[
                  { name: "Multinomial Naive Bayes", data: bayesMetrics, icon: "spark", color: "blue" },
                  { name: "Neural Network", data: neuralMetrics, icon: "activity", color: "violet" },
                ].map((item) => (
                  <article className={`panel model-card ${item.color}`} key={item.name}>
                    <div className="model-card-heading">
                      <div className="panel-icon"><Icon name={item.icon} /></div>
                      <div><h3>{item.name}</h3><p>{item.color === "blue" ? "Fast, interpretable baseline" : "Learns nonlinear patterns"}</p></div>
                    </div>
                    {item.data ? metricLabels.map(([key, label]) => (
                      <div className="metric-row" key={key}>
                        <span>{label}</span>
                        <div className="metric-bar"><i style={{ width: `${item.data[key] * 100}%` }} /></div>
                        <strong>{formatPercent(item.data[key])}</strong>
                      </div>
                    )) : <p className="missing-metrics">Train the models to see their test metrics.</p>}
                  </article>
                ))}
              </div>
              {metrics?.note && <p className="metrics-footnote"><Icon name="info" size={15} />{metrics.note}</p>}
            </>
          )}
        </section>
      </main>

      <footer className="footer">
        <a className="footer-brand" href="#"><span className="brand-mark"><Icon name="shield" size={18} /></span>signal<span>.</span></a>
        <p>Built for learning, not as the only line of defense. A prediction can be wrong—use your judgment.</p>
        <a href="#scanner" className="back-top">Back to scanner ↑</a>
      </footer>
    </div>
  );
}

export default App;
