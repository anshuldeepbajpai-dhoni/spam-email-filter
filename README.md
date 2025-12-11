# Spam Email Filter – Mini Project

## Overview

This mini project implements a simple **spam email classifier** using
Machine Learning (Naive Bayes) in Python. It trains on a dataset of
emails labeled as `spam` or `ham` (not spam), and then can predict
whether a new email is spam or not.

## Project Structure

- `data/sample_emails.csv` – Sample training data (you can add more).
- `models/` – Trained model will be saved here.
- `src/preprocess.py` – Text cleaning and preprocessing.
- `src/train_model.py` – Script to train and save the model.
- `src/predict_cli.py` – Command-line tool to predict spam vs ham.
- `requirements.txt` – Python dependencies.

## Setup Instructions

1. **Create virtual environment (optional but recommended)**
<!-- 
   ```bash
   cd spam_email_filter

   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On Linux / macOS
   python3 -m venv venv
   source venv/bin/activate -->