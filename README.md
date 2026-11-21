<p align="center">
  <img width="1050" alt="SMS spam detection dashboard" src="https://github.com/user-attachments/assets/14fa7cb7-c76e-4f52-bad7-4a8a8eda7552" />
</p>

# SMS Spam Detection Lab

A complete educational project for detecting spam SMS messages using two different machine-learning approaches:

- **Multinomial Naive Bayes** — uses Bayes' theorem, token probabilities, and per-word evidence.
- **PyTorch neural network** — learns patterns from TF-IDF text vectors.

Both models classify a message as **spam** or **ham** (legitimate). The project includes model training, evaluation, visualizations, an interactive React dashboard, a FastAPI API, and an original Streamlit interface.

## Table of Contents

- [Screenshots](#screenshots)
- [Features](#features)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Train the Models](#train-the-models)
- [Run the React Dashboard](#run-the-react-dashboard)
- [Run the Streamlit Dashboard](#run-the-streamlit-dashboard)
- [Evaluation Metrics](#evaluation-metrics)
- [Dataset](#dataset)
- [Limitations](#limitations)

## Screenshots

<p align="center">
  <img width="1050" alt="SMS spam detection dashboard" src="https://github.com/user-attachments/assets/14fa7cb7-c76e-4f52-bad7-4a8a8eda7552" />
</p>

<p align="center">
  <img width="1037" alt="Spam detection results" src="https://github.com/user-attachments/assets/ccdcce1c-ec31-41ab-b38f-0e32ddc6d9f1" />
</p>

## Features

- Downloads and loads the UCI SMS Spam Collection automatically.
- Falls back to a small synthetic dataset for offline training.
- Splits data into 70% training, 15% validation, and 15% testing using a stratified split.
- Implements Multinomial Naive Bayes with Laplace smoothing.
- Provides token-level explanations showing which words influence each prediction.
- Trains a PyTorch MLP: TF-IDF → 128 neurons → 64 neurons → sigmoid output.
- Stops neural-network training early when validation loss stops improving.
- Calculates accuracy, precision, recall, F1, ROC-AUC, average precision, and confusion matrices.
- Creates visualizations for class balance, model comparison, ROC curves, training history, and token evidence.
- Provides an adjustable spam threshold in the React dashboard.
- Offers predictions through FastAPI.

## How It Works

### 1. Text preprocessing and TF-IDF

Messages are converted into numerical features. Naive Bayes uses word and phrase frequencies, while the neural network receives TF-IDF vectors.

### 2. Model training

The training script trains both classifiers on the training split. The validation split is used for neural-network early stopping and model tuning.

### 3. Evaluation

The held-out test split measures how accurately each model identifies spam and ham messages.

### 4. Prediction

A prediction API and two dashboards accept text entered by the user and return a classification score.

## Project Structure

```text
sms-spam-detection/
├── api.py                         # FastAPI prediction API and built dashboard server
├── app.py                         # Original Streamlit dashboard
├── frontend/                      # React + Vite dashboard
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # Python project configuration
├── data/
│   └── demo_sms.csv               # Synthetic fallback dataset
├── docs/
│   └── MATHS_AND_ALGORITHM.md     # Theory and numerical Bayes example
├── models/                        # Saved trained model artifacts
├── outputs/
│   ├── metrics.json               # Test metrics and dataset information
│   ├── test_predictions.csv       # Predictions for the test split
│   ├── training_history.csv       # Neural-network training history
│   └── plots/                     # Generated visualizations
└── src/spamdetector/
    ├── bayes.py                   # Naive Bayes implementation
    ├── data.py                    # Dataset loading and download
    ├── models.py                  # PyTorch neural-network architecture
    ├── train.py                   # Training and evaluation pipeline
    └── visualize.py               # Plot generation
```

## Installation

### Windows 11 / PowerShell

Open PowerShell in the project folder, then run:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If script activation is blocked, use the virtual-environment Python directly:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Train the Models

From the project folder, run:

```powershell
python train.py
```

The loader checks `data/SMSSpamCollection` first. If that file is absent, it attempts to download the UCI ZIP. If the download fails, it trains with `data/demo_sms.csv` and prints a fallback warning.

### Offline demo

```powershell
python train.py --demo
```

### Change neural-network epochs

```powershell
python train.py --epochs 45
```

Early stopping may finish training before the maximum number of epochs is reached.

## Run the React Dashboard

### Step 1: Start the API

Open a terminal and run:

```powershell
uvicorn api:app --reload
```

The API is available at `http://127.0.0.1:8000`.

### Step 2: Start the frontend

Open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the URL shown by Vite, usually `http://localhost:5173`.

### Serve the dashboard and API together

```powershell
cd frontend
npm install
npm run build
cd ..
uvicorn api:app
```

Then open `http://127.0.0.1:8000`.

## Run the Streamlit Dashboard

After training is complete:

```powershell
streamlit run app.py
```

The Streamlit app allows you to:

- Enter or paste an SMS message.
- Select Naive Bayes or the neural network.
- Adjust the decision threshold.
- View the predicted spam score.
- Inspect Bayes token contributions.
- View test metrics and saved plots.

## Evaluation Metrics

- **Accuracy** — Percentage of test messages classified correctly.
- **Precision** — Percentage of messages predicted as spam that are actually spam.
- **Recall** — Percentage of actual spam messages detected by the model.
- **F1 score** — Balance between precision and recall.
- **ROC-AUC** — Ability to rank spam above ham across all thresholds.
- **Average precision** — Summary of precision across recall levels.
- **Confusion matrix** — Correct and incorrect predictions grouped by actual and predicted class.

A high accuracy score can hide poor spam detection when spam is uncommon. Compare precision, recall, F1, ROC-AUC, and the confusion matrix together.

## Bayes Calculation

For a message `M` and spam class `S`:

\[
P(S\mid M)=\frac{P(M\mid S)P(S)}{P(M\mid S)P(S)+P(M\mid H)P(H)}
\]

Multinomial Naive Bayes treats the message as a set of token counts:

\[
P(M\mid S)\approx\prod_{i=1}^{V}P(w_i\mid S)^{n_i}
\]

Laplace smoothing prevents unseen tokens from receiving a zero probability:

\[
P(w_i\mid S)=\frac{\text{count}(w_i,S)+\alpha}{N_S+\alpha V}
\]

The implementation performs these calculations in log space. See [`docs/MATHS_AND_ALGORITHM.md`](docs/MATHS_AND_ALGORITHM.md) for a worked numerical example.

## Dataset

The project attempts to use the UCI SMS Spam Collection:

Almeida, T. & Hidalgo, J. (2011). *SMS Spam Collection* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84

The dataset contains 5,574 English SMS messages and is licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

Demo-mode metrics are generated from a synthetic dataset. They validate the pipeline but should not be presented as real benchmark results.

## Limitations

- The dataset is historical and may not represent modern spam messages.
- It mainly contains English SMS text, so language and regional patterns may not generalize.
- Naive Bayes assumes that tokens are conditionally independent.
- Neural-network output scores may need calibration before production use.
- Retrain and evaluate on recent, representative, and properly licensed data before deployment.
- A prediction is a model estimate, not proof that a message is malicious. Keep human review and an appeal or recovery process.
