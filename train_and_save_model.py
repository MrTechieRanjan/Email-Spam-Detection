import pandas as pd
import string
import nltk
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords

nltk.download('stopwords')

# Clean function
def clean_text(text):
    text = ''.join([c for c in text if c not in string.punctuation])
    text = text.lower()
    words = text.split()
    return ' '.join([word for word in words if word not in stopwords.words('english')])

# Load dataset (example file)
df = pd.read_csv('emails.csv')  # Your dataset file here
df['cleaned_text'] = df['text'].apply(clean_text)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Vectorize and train
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['label_num']
model = MultinomialNB()
model.fit(X, y)

# Save model & vectorizer
joblib.dump(model, 'spam_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
