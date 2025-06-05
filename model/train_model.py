# model/train_model.py
import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

SAFE_TERMS = ["please", "content" ,"video", "more", "lmao", "jk","explode","fav","dare","kinda"]

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    cleaned = [word for word in words if word not in stop_words or word in SAFE_TERMS]
    return ' '.join(cleaned)

stop_words = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during", "before", "after", "above",
    "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
}

# Load and clean dataset
df = pd.read_csv("D:/BINUS/Spam-Detector-ML/Youtube-Spam-Dataset.csv")  # change to your actual CSV
df = df[['CONTENT', 'CLASS']]
df['CONTENT'] = df['CONTENT'].astype(str).apply(preprocess)

trusted = pd.DataFrame({
    'CONTENT': [
        "this explanation didn’t click for me",
        "this video was really helpful, thanks!",
        "bro's so rich, he use that vbux only for sabrina skin",
        "minecraft is free dawg"
    ],
    'CLASS': [0, 0, 0, 0]
})
trusted['cleaned'] = trusted['CONTENT'].apply(preprocess)
df = pd.concat([df, trusted], ignore_index=True)

# Balance the dataset
spam_df = df[df['CLASS'] == 1]
ham_df = df[df['CLASS'] == 0]
if len(spam_df) > len(ham_df):
    ham_df = resample(ham_df, replace=True, n_samples=len(spam_df))
else:
    spam_df = resample(spam_df, replace=True, n_samples=len(ham_df))
df_balanced = pd.concat([ham_df, spam_df])

# Split and train
X_train, X_test, y_train, y_test = train_test_split(
    df_balanced['CONTENT'], df_balanced['CLASS'], test_size=0.2, random_state=42
)
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Accuracy
accuracy = accuracy_score(y_test, model.predict(X_test_vec))
print("Model accuracy:", accuracy)

# Save model and vectorizer
joblib.dump(model, "model/spam_classifier.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")
joblib.dump(accuracy, "model/accuracy.pkl")

from sklearn.metrics import classification_report
print(classification_report(y_test, model.predict(X_test_vec)))
