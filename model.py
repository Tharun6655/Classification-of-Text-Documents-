import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv('./data/synthetic_text_data.csv')

X = data['text']
y = data['label']

print("Samples:", len(X))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42,
)

vectorizer =  TfidfVectorizer()

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

model = MultinomialNB(alpha=1.0)
model.fit(X_train_vectorized, y_train)

y_pred = model.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

class_labels=np.unique(y_test)

print(f'Accuracy: {accuracy * 100:.2f}%')
print("Confusion Matrix:")
print(conf_matrix) 


plt.figure(figsize=(8,6))

sns.heatmap(
    conf_matrix,
    annot=True,    
    fmt='d',        
    cmap='Blues',   
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap")
plt.show()

user_input = input('enter the text')

user_input_vectorized = vectorizer.transform([user_input])
predicted_label = model.predict(user_input_vectorized)
print(f"The input text belongs to the '{predicted_label[0]}' category.") 
