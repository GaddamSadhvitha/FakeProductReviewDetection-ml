from flask import Flask, render_template, request
from keras.models import load_model
import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('model.h5')  # Ensure the path matches the location of your model file

# Load the saved TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

# Download necessary NLTK data files
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize the Lemmatizer and stopwords list
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess_text(text):
    """
    Preprocess the input text: lowercase, remove special characters, lemmatize, and remove stopwords.
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    words = text.split()  # Tokenize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatize and remove stopwords
    return ' '.join(words)

# Route for the home page
@app.route('/')
def home():
    """
    Render the homepage with the input form.
    """
    return render_template('index.html')  # Renders the HTML form

# Route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle review submissions, preprocess text, and predict if it's fake or real.
    """
    if request.method == 'POST':
        review = request.form['review']  # Get the review from the form
        if review.strip():  # Ensure the input is not empty
            # Preprocess the input review
            processed_review = preprocess_text(review)
            
            # Debug: Log the processed review
            print(f"Processed review: {processed_review}")
            
            # Vectorize the processed text using the loaded TF-IDF vectorizer
            vectorized_review = tfidf_vectorizer.transform([processed_review])
            
            # Debug: Log the vectorized review
            print(f"Vectorized review: {vectorized_review.toarray()}")
            
            # Get the prediction from the model
            prediction = model.predict(vectorized_review)[0][0]
            
            # Debug: Log the raw prediction score
            print(f"Raw model prediction: {prediction}")
            
            # Determine the label based on the prediction threshold
            label = "Real" if prediction > 0.55 else "Fake"  # Adjust threshold if necessary
            
            # Render the result on the webpage
            return render_template('index.html', prediction=label, review=review)
    
    # Render the homepage if no input is provided
    return render_template('index.html', prediction=None, review="")

if __name__ == "__main__":
    app.run(debug=True)  # Start the Flask server
