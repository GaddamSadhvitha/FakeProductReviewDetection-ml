import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.regularizers import l2  # Regularizer to avoid overfitting
import pickle

# Load the dataset
df = pd.read_csv("fake_reviews_dataset.csv")

# Download necessary NLTK data files
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize the Lemmatizer and stopwords list
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Preprocessing function for the text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    words = text.split()  # Tokenize the text
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Remove stopwords and lemmatize
    return ' '.join(words)

# Preprocess the reviews
processed_reviews = df['text'].apply(preprocess_text)

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

# Apply the vectorizer to the preprocessed reviews
X_tfidf = tfidf_vectorizer.fit_transform(processed_reviews)

# Save the trained TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'wb') as file:
    pickle.dump(tfidf_vectorizer, file)

# Labels
y = df['label']  # Replace 'label' with your label column

# Split the data into training and validation sets
X_train_tfidf, X_val_tfidf, y_train, y_val = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential()

# Input layer with L2 regularization to prevent overfitting
model.add(Dense(512, activation='relu', input_dim=X_tfidf.shape[1], kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))  # Dropout layer with 50% rate

# Hidden layers with L2 regularization
model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))  # Dropout layer with 50% rate

# Output layer
model.add(Dense(1, activation='sigmoid'))  # Sigmoid for binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Early stopping to prevent overfitting (stop training if validation loss doesn't improve for 2 epochs)
early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# Train the model
model.fit(X_train_tfidf, y_train, epochs=5, batch_size=32, validation_data=(X_val_tfidf, y_val), callbacks=[early_stopping])

# Save the trained model
model.save("model.h5")

# Predict using the trained model
all_predictions = model.predict(X_tfidf)

# Map predictions to 'Real' or 'Fake'
df['predicted_label'] = ['Real' if pred > 0.5 else 'Fake' for pred in all_predictions]

# Save the output to a CSV file with the updated label
df.to_csv('predicted_reviews_all_with_labels.csv', index=False)
