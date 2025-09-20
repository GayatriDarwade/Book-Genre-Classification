import pickle
import re
import os
from flask import Flask, request, render_template
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
import nltk

# =======================
# NLTK Local Corpora Setup
# =======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPORA_DIR = os.path.join(BASE_DIR, "corpora")

# Add local corpora folder to NLTK path
if CORPORA_DIR not in nltk.data.path:
    nltk.data.path.append(CORPORA_DIR)

# Load stopwords
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    raise RuntimeError(f"Stopwords not found! Check {CORPORA_DIR}/stopwords")

# Load WordNet
try:
    nltk.corpus.wordnet.ensure_loaded()
except LookupError:
    raise RuntimeError(f"WordNet not found! Check {CORPORA_DIR}/wordnet")

# =======================
# Text Preprocessing
# =======================
def cleantext(text):
    text = re.sub("'\''", "", text)
    text = re.sub("[^a-zA-Z]", " ", text)
    return ' '.join(text.split()).lower()

def removestopwords(text):
    return ' '.join([w for w in text.split() if w not in stop_words])

def lemmatizing(text):
    lemma = WordNetLemmatizer()
    return ' '.join([lemma.lemmatize(w) for w in text.split()])

def stemming(text):
    stemmer = PorterStemmer()
    return ' '.join([stemmer.stem(w) for w in text.split()])

# =======================
# Model Prediction
# =======================
def test(text, model, tfidf_vectorizer):
    text = cleantext(text)
    text = removestopwords(text)
    text = lemmatizing(text)
    text = stemming(text)
    vector = tfidf_vectorizer.transform([text])
    pred = model.predict(vector)
    mapping = {0: 'Fantasy', 1: 'Science Fiction', 2: 'Crime Fiction',
               3: 'Historical novel', 4: 'Horror', 5: 'Thriller'}
    return mapping[pred[0]]

# =======================
# Load Model & Vectorizer
# =======================
with open(os.path.join(BASE_DIR, "bookgenremodel.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "tfdifvector.pkl"), "rb") as f:
    tfidf_vectorizer = pickle.load(f)

# =======================
# Flask App
# =======================
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        text = request.form.get("summary", "")
        prediction = test(text, model, tfidf_vectorizer)
        return render_template("index.html", genre=prediction, text=text[:100], showresult=True)
    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
