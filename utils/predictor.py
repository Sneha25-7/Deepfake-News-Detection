import joblib

model = joblib.load("models/fake_news.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def predict_news(news):

    vector = vectorizer.transform([news])

    prediction = model.predict(vector)[0]

    confidence = model.decision_function(vector)[0]

    if prediction == 1:
        result = "REAL NEWS"
    else:
        result = "FAKE NEWS"

    return result, abs(round(confidence * 10, 2))