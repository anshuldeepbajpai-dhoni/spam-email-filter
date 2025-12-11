from flask import Flask, request, render_template
import joblib
from src.config import MODEL_PATH

app = Flask(__name__)

# Load model
model = joblib.load(MODEL_PATH)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["email"]
    pred = model.predict([text])[0]
    result = "SPAM" if pred == 1 else "NOT SPAM"
    return render_template("result.html", email=text, prediction=result)


if __name__ == "__main__":
    app.run(debug=True)
