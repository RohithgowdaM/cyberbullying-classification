# Import necessary libraries
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import nltk

# Download required NLTK resources
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset
file_path = './cyberbullying_tweets(ML).csv'
data = pd.read_csv(file_path)

# Define a leetspeak dictionary for normalization
leet_dict = {
    '@': 'a', '$': 's', '0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's', '7': 't', '!': 'i'
}

# Function to normalize leetspeak
def normalize_leetspeak(text):
    for symbol, replacement in leet_dict.items():
        text = text.replace(symbol, replacement)
    return text

# Enhanced preprocessing function
def preprocess_text_advanced(text):
    # Initialize Lemmatizer and define stop words
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Normalize leetspeak
    text = normalize_leetspeak(text)
    
    # Remove URLs and mentions
    text = re.sub(r'http\S+|@\w+', '', text)
    
    # Tokenize and lowercase
    tokens = word_tokenize(text.lower())
    
    # Remove punctuation and stop words
    tokens = [re.sub(r'[^a-zA-Z]', '', word) for word in tokens if word not in stop_words]
    
    # Lemmatize the tokens
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens if word]
    
    return " ".join(lemmatized_tokens)

# Apply the enhanced preprocessing function
data['processed_tweet'] = data['tweet_text'].apply(preprocess_text_advanced)

# Transform tweets into TF-IDF vectors
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(data['processed_tweet']).toarray()
y = data['cyberbullying_type']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=100)
}

# Train and evaluate models
best_model_name = None
best_accuracy = 0

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluation Metrics
    print(f"\n{name} - Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print(f"\n{name} - Classification Report:")
    print(classification_report(y_test, y_pred))
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{name} - Accuracy: {accuracy:.2f}\n")
    
    # Check for best model
    if accuracy > best_accuracy:
        best_model_name = name
        best_accuracy = accuracy

print(f"The best model is {best_model_name} with an accuracy of {best_accuracy:.2f}.")
