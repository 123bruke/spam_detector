<img width="1037" height="560" alt="Screenshot 2026-11-19 231733" src="https://github.com/user-attachments/assets/ccdcce1c-ec31-41ab-b38f-0e32ddc6d9f1" />
# SMS Spam Detection Lab: Bayes' Theorem + Neural Network

A complete educational Python project that classifies a text message as **spam** or **ham** (legitimate). It trains and compares two models:

1. **Multinomial Naive Bayes** — applies Bayes' theorem to class priors and word/phrase likelihoods. It also exposes token-level evidence that pushes a prediction toward spam or ham.
2. **Neural Network (PyTorch MLP)** — learns patterns from TF-IDF text vectors using a trainable feed-forward neural network.

The project evaluates both models on a held-out test split and produces visualizations using **Seaborn and Matplotlib**. A React dashboard lets you test your own messages interactively. The original Streamlit dashboard is also available.

> **Dataset note:** The project attempts to download the official UCI SMS Spam Collection automatically. The dataset has 5,574 messages and is licensed CC BY 4.0. If the download is unavailable, the project falls back to a small, synthetic, balanced demo dataset so that you can test the code offline. Demo metrics are only a pipeline smoke test and must not be presented as benchmark results.

## Features

- Real-data loader for the UCI SMS Spam Collection, with offline demo fallback.
- Stratified train / validation / test split (approximately 70% / 15% / 15%).
- Bayes' theorem implemented with Multinomial Naive Bayes and Laplace smoothing.
- Manual log-posterior calculation and per-token spam-vs-ham log-odds explanation.
- Trainable PyTorch neural network: TF-IDF → Dense(128) → Dense(64) → output sigmoid.
- Early stopping using validation loss.
- Accuracy, precision, recall, F1, ROC-AUC, average precision, confusion matrices, and ROC curves.
- Seaborn/Matplotlib visualizations: class balance, model comparison, confusion matrices, ROC, training curves, and Bayes token evidence.
- Interactive React prediction dashboard with adjustable spam threshold, token evidence, and model metrics.
- FastAPI backend for predictions from the trained models.
- Original Streamlit prediction dashboard remains available.

## Project structure

```text
sms-spam-detection/
├── api.py                         # FastAPI backend and built React app server
├── app.py                         # Streamlit interface
├── frontend/                      # React + Vite dashboard
├── requirements.txt
├── pyproject.toml
├── data/
│   └── demo_sms.csv               # Small synthetic offline smoke-test dataset
├── docs/
│   └── MATHS_AND_ALGORITHM.md     # Theory and worked Bayes calculation
├── models/                        # Trained model artifacts (created by training)
├── outputs/
│   ├── metrics.json               # Test metrics and dataset source
│   ├── test_predictions.csv       # Held-out predictions
│   ├── training_history.csv       # Neural-network epoch history
│   └── plots/                     # Seaborn/Matplotlib figures
└── src/spamdetector/
    ├── bayes.py                   # Bayes / MultinomialNB implementation
    ├── data.py                    # Dataset download and loading
    ├── models.py                  # PyTorch MLP architecture
    ├── train.py                   # End-to-end training/evaluation
    └── visualize.py               # Visualization functions
```

## 1. Installation (Windows 11 / PowerShell)

Open PowerShell inside the project folder:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation for this terminal, you can use the environment's Python directly:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2. Train both models

```powershell
python train.py
```

The data loader first checks for `data/SMSSpamCollection`. If it is not present, it tries to download the UCI ZIP. If the download fails, it trains on `data/demo_sms.csv` and prints a warning.

To explicitly run the small offline demo:

```powershell
python train.py --demo
```

To change the maximum number of neural-network epochs:

```powershell
python train.py --epochs 45
```

Early stopping may finish before the maximum epoch count.

## 3. Launch the React dashboard

In one PowerShell terminal, start the prediction API:

```powershell
uvicorn api:app --reload
```

In a second terminal, install the React dependencies and start the frontend:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite (usually `http://localhost:5173`). The dashboard calls the API at `http://127.0.0.1:8000`.

To serve the built React dashboard and API from one local server instead:

```powershell
cd frontend
npm install
npm run build
cd ..
uvicorn api:app
```

Then open `http://127.0.0.1:8000`.

## 4. Launch the original Streamlit dashboard

After training succeeds:

```powershell
streamlit run app.py
```

The Streamlit dashboard lets you paste a message, select either classifier, change the decision threshold, view the predicted spam score, and inspect the Bayes token contributions. It also displays test metrics and plots saved by the training script.

## 5. Understanding the calculations

### Bayes' theorem

For a message `M` and class `S = spam`:

\[
P(S\mid M)=\frac{P(M\mid S)P(S)}{P(M)}
\]

With two classes:

\[
P(S\mid M)=\frac{P(M\mid S)P(S)}{P(M\mid S)P(S)+P(M\mid H)P(H)}
\]

where `H` means ham. Multinomial Naive Bayes models a message as a set of token counts and uses the conditional-independence approximation:

\[
P(M\mid S)\approx\prod_{i=1}^{V}P(w_i\mid S)^{n_i}
\]

`n_i` is the count of token `w_i` in the message. Laplace smoothing prevents unseen tokens from receiving zero probability:

\[
P(w_i\mid S)=\frac{count(w_i,S)+\alpha}{N_S+\alpha V}
\]

The implementation works in log space to avoid multiplying many tiny probabilities:

\[
Score(S)=\log P(S)+\sum_i n_i\log P(w_i\mid S)
\]

The posterior is obtained by normalizing the class scores:

\[
P(S\mid M)=\frac{e^{Score(S)}}{e^{Score(S)}+e^{Score(H)}}
\]

See [`docs/MATHS_AND_ALGORITHM.md`](docs/MATHS_AND_ALGORITHM.md) for a numerical worked example.

### Neural network

The neural network receives a TF-IDF vector. It learns weights using backpropagation and binary cross-entropy loss. The final sigmoid output is a **spam score**. Since the training loss uses a positive-class weight to account for class imbalance, the score should not automatically be interpreted as a perfectly calibrated real-world probability.

## 6. How to interpret evaluation metrics

- **Accuracy:** fraction of all test messages classified correctly.
- **Precision:** among messages predicted as spam, the fraction that really are spam.
- **Recall:** among actual spam messages, the fraction detected by the model.
- **F1:** harmonic mean of precision and recall.
- **ROC-AUC:** ranking ability across possible thresholds; 1 is ideal and 0.5 is random ranking.
- **Confusion matrix:** true ham, false spam, missed spam, and correctly detected spam counts.

A low spam prevalence can make accuracy look high even when spam recall is poor. Examine precision, recall, F1 and the confusion matrix together.

## Dataset attribution

Almeida, T. & Hidalgo, J. (2011). *SMS Spam Collection* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84. Dataset license: Creative Commons Attribution 4.0 International (CC BY 4.0).

## Limitations and responsible use

- The UCI collection contains English SMS messages. Its patterns may not generalize to other languages, countries, or modern messaging platforms.
- The dataset is historical; spam tactics change. Retrain and evaluate on recent, representative, consented data before real deployment.
- The Naive Bayes model assumes token conditional independence, which is a simplifying assumption.
- The neural network's output score may require probability calibration before production use.
- A prediction is not proof that a message is malicious. Keep human review and a way to recover incorrectly flagged messages.
