import joblib
import gradio as gr
import re

# Load model, vectorizer, accuracy
model = joblib.load("model/spam_classifier.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")
accuracy = joblib.load("model/accuracy.pkl")

stop_words = {...}  # Use the same set as in training
SAFE_TERMS = ["please", "content" ,"video", "more", "lmao", "jk","explode","fav","dare","kinda"]

# Preprocess function (same as your training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    return ' '.join([word for word in words if word not in stop_words or word in SAFE_TERMS])

# Prediction function
def predict_spam(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    pred = model.predict(vectorized)[0]
    label = "Spam" if pred == 1 else "Not Spam"
    return f"Prediction: {label} (Accuracy: {round(accuracy*100, 2)}%)"

# Gradio UI
demo = gr.Interface(
    fn=predict_spam,
    inputs=gr.Textbox(lines=3, placeholder="Enter your comment here..."),
    outputs="text",
    title="Spam Comment Detector",
    description="Enter a comment and see if it's spam. Model trained on YouTube spam dataset."
)

demo.launch()
