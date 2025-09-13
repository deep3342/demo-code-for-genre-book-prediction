# demo-code-for-genre-book-prediction
# Step 1: Import libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Step 2: Example dataset (you can replace this with a bigger dataset)
data = {
    "text": [
        "A love story between two people in Paris", 
        "A detective solves a mysterious murder case",
        "Spaceships battle in an intergalactic war",
        "A young wizard fights against dark magic",
        "A stand-up comedian struggles with family life"
    ],
    "genre": [
        "Romance", 
        "Mystery", 
        "Sci-Fi", 
        "Fantasy", 
        "Comedy"
    ]
}

df = pd.DataFrame(data)

# Step 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["genre"], test_size=0.2, random_state=42)

# Step 4: Convert text into TF-IDF features
vectorizer = TfidfVectorizer(stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 5: Train Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Step 6: Predictions
y_pred = model.predict(X_test_tfidf)

# Step 7: Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 8: Try with new input
new_text = ["A spaceship travels through galaxies to fight aliens"]
new_text_tfidf = vectorizer.transform(new_text)
prediction = model.predict(new_text_tfidf)
print("\nPredicted Genre:", prediction[0])
