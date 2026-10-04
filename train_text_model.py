import os
import joblib
import pandas as pd

print("Current Folder:", os.getcwd())
print("Dataset Exists:", os.path.exists("dataset/WELFake_Dataset.csv"))


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Dataset path
DATASET_PATH = "dataset/WELFake_Dataset.csv"

# Check if dataset exists
if not os.path.exists(DATASET_PATH):
    print(f"Error: Dataset not found at {DATASET_PATH}")
    exit()

# Load dataset
data = pd.read_csv(DATASET_PATH)

print("Dataset Loaded Successfully")
print(data.head())

# Keep only required columns
data = data[['title', 'text', 'label']]

# Replace missing values
data['title'] = data['title'].fillna('')
data['text'] = data['text'].fillna('')

# Combine title and article text
data['content'] = data['title'] + " " + data['text']

# Features and labels
X = data['content']
y = data['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.7,
    max_features=50000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = PassiveAggressiveClassifier(max_iter=1000)

model.fit(X_train_vec, y_train)

# Predictions
y_pred = model.predict(X_test_vec)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save model
joblib.dump(model, "models/fake_news.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\nModel Saved Successfully!")
print("Saved files:")
print("models/fake_news.pkl")
print("models/vectorizer.pkl")