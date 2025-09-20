import os
import re
import pickle
from flask import Flask, request, render_template
from nltk.stem import PorterStemmer, WordNetLemmatizer

# --- Paths ---
BASE_DIR = os.path.dirname(__file__)
CORPORA_DIR = os.path.join(BASE_DIR, "corpora")
STOPWORDS_DIR = os.path.join(CORPORA_DIR, "stopwords")
WORDNET_DIR = os.path.join(CORPORA_DIR, "wordnet")

# --- Load Stopwords Manually ---
if not os.path.exists(STOPWORDS_DIR):
    raise RuntimeError(f"Stopwords folder not found at {STOPWORDS_DIR}")

stop_words = set()
for file in os.listdir(STOPWORDS_DIR):
    filepath = os.path.join(STOPWORDS_DIR, file)
    with open(filepath, "r", encoding="utf-8") as f:
        stop_words.update([line.strip() for line in f])

# --- Configure NLTK to use Local WordNet ---
import nltk
nltk.data.path.append(WORDNET_DIR)

# Test WordNet
try:
    WordNetLemmatizer()
except LookupError:
    raise RuntimeError(f"WordNet not found in {WORDNET_DIR}")

# --- Text Processing Functions ---
def cleantext(text):
    text = re.sub("'\''", "", text)
    text = re.sub("[^a-zA-Z]", " ", text)
    text = ' '.join(text.split())
    return text.lower()

def removestopwords(text):
    return ' '.join([word for word in text.split() if word not in stop_words])

def lemmatizing(text):
    lemma = WordNetLemmatizer()
    return ' '.join([lemma.lemmatize(word) for word in text.split()])

def stemming(text):
    stemmer = PorterStemmer()
    return ' '.join([stemmer.stem(word) for word in text.split()])

# --- Prediction Function ---
def test(text, model, tfidf_vectorizer):
    text = cleantext(text)
    text = removestopwords(text)
    text = lemmatizing(text)
    text = stemming(text)
    text_vector = tfidf_vectorizer.transform([text])
    predicted = model.predict(text_vector)
    newmapper = {0: 'Fantasy', 1: 'Science Fiction', 2: 'Crime Fiction',
                 3: 'Historical novel', 4: 'Horror', 5: 'Thriller'}
    return newmapper[predicted[0]]

# --- Load Model and Vectorizer ---
with open(os.path.join(BASE_DIR, 'bookgenremodel.pkl'), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, 'tfdifvector.pkl'), 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# --- Flask App ---
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form.get("summary", "")
        if text.strip() == "":
            return render_template('index.html', showresult=False, message="Please enter some text!")
        prediction = test(text, model, tfidf_vectorizer)
        return render_template('index.html', genre=prediction, text=text[:100], showresult=True)
    return render_template('index.html', showresult=False)

# --- Run App ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
