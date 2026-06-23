# Classification of Text Documents using Naive Bayes

An end-to-end Natural Language Processing (NLP) and Machine Learning project that classifies text documents into predefined categories using the Naive Bayes algorithm. This implementation utilizes Bayes' Theorem to calculate the likelihood of a document belonging to a specific class based on the distribution of words within the text.

## 📌 Project Overview
Text classification is a foundational task in modern AI, powering systems like spam detection, sentiment analysis, and automated news tagging. This project showcases how to preprocess raw text, convert textual features into numerical vectors using techniques like TF-IDF or Count Vectorization, and apply a Multinomial Naive Bayes classifier for quick and highly scalable categorization.

## 🗂️ Project Structure
```text
Text_Document_Classification/
│
├── Data/
│   └── documents.csv                  # Dataset containing raw text and corresponding labels
│
├── notebooks/                         # (Optional) Jupyter notebooks for text exploration
│
├── .gitignore                         # Prevents tracking virtual environment (venv/)
├── classify.py                        # Python script for text preprocessing, vectorization, and modeling
├── README.md                          # Project documentation
└── requirements.txt                   # List of Python dependencies (scikit-learn, pandas, nltk, etc.)
