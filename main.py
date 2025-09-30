from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request 
import joblib
import re

app = FastAPI()

templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="templates"), name="static") 

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

model = joblib.load("model/spam_classifier.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")
accuracy = joblib.load("model/accuracy.pkl")

stop_words = {...}  
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

@app.get("/")
def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict(request: CommentRequest):
    cleaned = preprocess(request.comment)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    label = "Spam" if int(prediction) == 1 else "Not Spam"
    return {"prediction": label, "accuracy": round(accuracy * 100, 2)}
