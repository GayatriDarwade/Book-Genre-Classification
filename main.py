import pickle
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
import re
import nltk
from flask import Flask, request, render_template
import os

# Add local corpora folder to NLTK path
nltk.data.path.append(os.path.join(os.path.dirname(__file__), "corpora"))

# Ensure stopwords and wordnet are accessible
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    raise RuntimeError("Stopwords not found in local corpora folder!")

try:
    WordNetLemmatizer()  # Just to trigger loading wordnet
except LookupError:
    raise RuntimeError("WordNet not found in local corpora folder!")

# Cleaning the text
def cleantext(text):
    text = re.sub("'\''", "", text)
    text = re.sub("[^a-zA-Z]", " ", text)
    text = ' '.join(text.split())
    return text.lower()

# Removing stopwords
def removestopwords(text):
    removedstopword = [word for word in text.split() if word not in stop_words]
    return ' '.join(removedstopword)

# Lemmatizing
def lemmatizing(text):
    lemma = WordNetLemmatizer()
    return ' '.join([lemma.lemmatize(word) for word in text.split()])

# Stemming
def stemming(text):
    stemmer = PorterStemmer()
    return ' '.join([stemmer.stem(word) for word in text.split()])

# Test model
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

# Load model and vectorizer
with open('bookgenremodel.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfdifvector.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'POST':
        text = request.form["summary"]
        prediction = test(text, model, tfidf_vectorizer)
        return render_template('index.html', genre=prediction, text=str(text)[:100], showresult=True)
    return render_template('index.html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
