import boto3
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
RAW_CSV = DATA_DIR / "sp500_close.csv"
MATRIX_PKL = DATA_DIR / "sp500_10yr_matrix.pkl"

def load_env():
    env = {}
    env_file = BASE_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env

if not RAW_CSV.exists():
    print("Downloading sp500_close.csv (92.8 MB) from S3...")
    env = load_env()
    s3 = boto3.client(
        "s3",
        aws_access_key_id=env["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=env["AWS_SECRET_ACCESS_KEY"],
        region_name=env.get("AWS_DEFAULT_REGION", "us-west-2")
    )
    s3.download_file("rawdatasp", "historicaldata/sp500_close.csv", str(RAW_CSV))
    print("Download complete.")

print("Building 10-year price matrix (2013-09-29 to 2023-09-29)...")
df = pd.read_csv(RAW_CSV, usecols=["ShareName", "Date", "Value"])
df = df[df["Date"] >= "2013-09-29"].dropna(subset=["Value"])
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
df = df.dropna(subset=["Value"])

matrix = df.pivot_table(index="Date", columns="ShareName", values="Value")
matrix.index = pd.to_datetime(matrix.index)
matrix = matrix.sort_index()
# Forward fill occasional missing trading days, drop columns with < 90% data
matrix = matrix.ffill()
min_days = int(len(matrix) * 0.90)
matrix = matrix.dropna(axis=1, thresh=min_days)

matrix.to_pickle(MATRIX_PKL)
print(f"Saved matrix: {matrix.shape[0]} trading days x {matrix.shape[1]} stocks -> {MATRIX_PKL}")
