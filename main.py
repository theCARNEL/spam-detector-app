from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import re

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Load artifacts
model = joblib.load("model/spam_classifier.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")
accuracy = joblib.load("model/accuracy.pkl")

stop_words = {...}  # Use the same set as in training
SAFE_TERMS = ["please", "content" ,"video", "more", "lmao", "jk","explode","fav","dare","kinda"]


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    return ' '.join([word for word in words if word not in stop_words or word in SAFE_TERMS])


def preprocess(text):
    text = re.sub(r"[^\w\s]", "", text.lower())
    return " ".join(word for word in text.split() if word not in stop_words)

class CommentRequest(BaseModel):
    comment: str

@app.post("/predict")
def predict(request: CommentRequest):
    cleaned = preprocess(request.comment)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    label = "Spam" if int(prediction) == 1 else "Not Spam"
    return {"prediction": label, "accuracy": round(accuracy * 100, 2)}
