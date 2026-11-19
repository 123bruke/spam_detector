"""Load the UCI SMS Spam Collection, with a small offline demo fallback."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen
# used to parsing the new conditions of file formates and streams and varibles and common terms and var 

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
UCI_FILE = DATA_DIR / "SMSSpamCollection"
DEMO_FILE = DATA_DIR / "demo_sms.csv"
DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def _read_uci(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["label", "text"],
        usecols=[0, 1],
        quoting=3,
        encoding="utf-8",
        on_bad_lines="skip",
    )
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df = df[df["label"].isin(["ham", "spam"])].copy()
    df["label_id"] = df["label"].map({"ham": 0, "spam": 1}).astype(int)
    return df[["label", "label_id", "text"]]


def load_dataset(force_demo: bool = False) -> tuple[pd.DataFrame, str]:
    """Return (dataframe, source_name). Download UCI data when it is not local."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not force_demo:
        if UCI_FILE.exists():
            data = _read_uci(UCI_FILE)
            source = "UCI SMS Spam Collection (local copy)"
        else:
            try:
                print("Downloading the UCI SMS Spam Collection ...")
                with urlopen(DATA_URL, timeout=25) as response:
                    archive_bytes = response.read()
                with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
                    member = next(
                        (name for name in archive.namelist() if name.endswith("SMSSpamCollection")),
                        None,
                    )
                    if member is None:
                        raise FileNotFoundError("SMSSpamCollection was not found in the UCI ZIP file")
                    UCI_FILE.write_bytes(archive.read(member))
                data = _read_uci(UCI_FILE)
                source = "UCI SMS Spam Collection (downloaded)"
            except Exception as exc:  # offline or temporary download failure
                print(f"WARNING: UCI dataset download failed: {exc}")
                print("Using the small synthetic demo dataset. Do not treat demo metrics as benchmark results.")
                data = pd.read_csv(DEMO_FILE)
                data["label_id"] = data["label"].map({"ham": 0, "spam": 1}).astype(int)
                source = "Synthetic demo dataset (offline fallback)"
    else:
        data = pd.read_csv(DEMO_FILE)
        data["label_id"] = data["label"].map({"ham": 0, "spam": 1}).astype(int)
        source = "Synthetic demo dataset (forced)"

    data["text"] = data["text"].astype(str).str.strip()
    data = data[data["text"].str.len() > 0]
    data = data.drop_duplicates(subset="text").reset_index(drop=True)
    if data["label_id"].nunique() != 2:
        raise ValueError("The dataset must contain both ham (0) and spam (1) classes.")
    return data, source
