from pathlib import Path

# Base directory of the project (spam_email_filter/)
BASE_DIR = Path(__file__).resolve().parents[1]

# Path to the CSV dataset
DATA_PATH = BASE_DIR / "data" / "sample_emails.csv"

# Path where the trained model will be saved
MODEL_PATH = BASE_DIR / "models" / "spam_classifier.joblib"
