import joblib
import string
from nltk.corpus import stopwords

# Clean function
def clean_text(text):
    text = ''.join([c for c in text if c not in string.punctuation])
    text = text.lower()
    words = text.split()
    return ' '.join([word for word in words if word not in stopwords.words('english')])

# Load model and vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Manual testing loop
while True:
    user_input = input("\nEnter your email text (or type 'exit'):\n")
    if user_input.lower() == 'exit':
        break
    cleaned = clean_text(user_input)
    vector = vectorizer.transform([cleaned])
    result = model.predict(vector)
    print("🟢 Not Spam" if result[0] == 0 else "🔴 Spam")
