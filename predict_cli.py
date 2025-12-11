# import joblib
# from .config import MODEL_PATH

# def load_model():
#     """
#     Load the ML model from the saved .pkl file.
#     """
#     try:
#         model = joblib.load(MODEL_PATH)
#         print(f"Model loaded successfully from: {MODEL_PATH}")
#         return model
#     except Exception as e:
#         print("Error loading model:", e)
#         exit()


# def interpret_prediction(pred):
#     """
#     Convert model output into human-readable text.
    
#     Handles:
#     - Numeric labels (0 = NOT SPAM, 1 = SPAM)
#     - String labels ("spam", "ham", etc.)
#     - Fallback for unknown types
#     """

#     # If model output is numeric (0 or 1)
#     if isinstance(pred, (int, float)):
#         if pred == 1:
#             return "SPAM"
#         else:
#             return "NOT SPAM"

#     # If model output is string
#     if isinstance(pred, str):
#         text = pred.strip().lower()   # safe
#         if "spam" in text:
#             return "SPAM"
#         else:
#             return "NOT SPAM"

#     # Unknown type fallback
#     return "NOT SPAM"


# def main():
#     model = load_model()

#     print("\n=== Spam Email Filter (CLI) ===")
#     print("Type/paste your email text below.")
#     print("Type 'quit' or 'exit' to stop.\n")

#     while True:
#         user_input = input("Email text> ").strip()

#         # Exit condition
#         if user_input.lower() in {"quit", "exit"}:
#             print("\nThank you for using the Spam Email Filter! 😊")
#             break

#         # Empty input handling
#         if user_input == "":
#             print("Please enter some text.\n")
#             continue

#         try:
#             raw_prediction = model.predict([user_input])[0]
#         except Exception as e:
#             print("Error during prediction:", e)
#             continue

#         # Convert raw output to readable label
#         final_label = interpret_prediction(raw_prediction)

#         print(f"Prediction: {final_label}\n")


# if __name__ == "__main__":
#     main()
#python -m src.train_model
#python -m src.predict_cli
import joblib
from .config import MODEL_PATH

def load_model():
    print(f"Model loaded successfully from: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    return model

def interpret(pred):
    """
    Your SVM model outputs:
      1 -> SPAM
      0 -> NOT SPAM
    """
    if pred == 1:
        return "SPAM"
    else:
        return "NOT SPAM"

def main():
    model = load_model()

    print("\n=== Spam Email Filter (CLI) ===")
    print("Type/paste your email text below.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        text = input("Email text> ").strip()

        if text.lower() in {"quit", "exit"}:
            print("\nThank you for using the Spam Email Filter! 😊")
            break

        pred = model.predict([text])[0]     # <-- SVM returns 0 or 1
        result = interpret(pred)

        print(f"Prediction: {result}\n")

if __name__ == "__main__":
    main()
