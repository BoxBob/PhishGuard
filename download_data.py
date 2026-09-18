"""
Fetches the raw datasets that preprocess.py expects in data/raw/:
  - phishtank.csv  <- PhishTank verified phishing feed
  - urlhaus.csv    <- URLhaus full malicious URL dump
  - tranco.csv     <- Tranco Top 1M legitimate domains

ISCX URL 2016 is not fetched here (Kaggle requires auth) - drop the
downloaded CSV in as data/raw/iscx.csv with 'URL' and 'type' columns.
"""
import io
import os
import zipfile
import requests

RAW_DIR = "data/raw"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

PHISHTANK_URL = "https://data.phishtank.com/data/online-valid.csv"
URLHAUS_URL = "https://urlhaus.abuse.ch/downloads/csv/"
TRANCO_URL = "https://tranco-list.eu/top-1m.csv.zip"


def download(url, dest_path, label):
    print(f"Downloading {label}...")
    response = requests.get(url, headers=HEADERS, timeout=60)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(response.content)
    size_kb = len(response.content) / 1024
    print(f" [+] Saved {label} to {dest_path} ({size_kb:.1f} KB)")


def download_zipped_csv(url, dest_path, label):
    print(f"Downloading {label}...")
    response = requests.get(url, headers=HEADERS, timeout=60)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        name = z.namelist()[0]
        with z.open(name) as src, open(dest_path, "wb") as dst:
            dst.write(src.read())
    print(f" [+] Saved {label} to {dest_path}")


if __name__ == "__main__":
    os.makedirs(RAW_DIR, exist_ok=True)

    download(PHISHTANK_URL, os.path.join(RAW_DIR, "phishtank.csv"), "PhishTank")
    download_zipped_csv(URLHAUS_URL, os.path.join(RAW_DIR, "urlhaus.csv"), "URLhaus")
    download_zipped_csv(TRANCO_URL, os.path.join(RAW_DIR, "tranco.csv"), "Tranco Top 1M")

    iscx_path = os.path.join(RAW_DIR, "iscx.csv")
    if os.path.exists(iscx_path):
        print(f" [+] ISCX dataset already present at {iscx_path}")
    else:
        print(f" [!] ISCX dataset missing - place the Kaggle CSV at {iscx_path} before running preprocess.py")
