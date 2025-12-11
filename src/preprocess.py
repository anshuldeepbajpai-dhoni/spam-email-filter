import re

def clean_text(text):
    text = text.lower()

    # Keep URLs (important)
    text = re.sub(r"http\S+|www\S+|\S+\.com", " url ", text)

    # Replace numbers with token
    text = re.sub(r"\d+", " number ", text)

    # Keep words, numbers, and URLs
    text = re.sub(r"[^a-zA-Z0-9:/._ ]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text
