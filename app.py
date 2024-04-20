import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
# Clean Datasets

from nltk.stem.porter import PorterStemmer
from collections import Counter

ps = PorterStemmer()
wnl = nltk.stem.WordNetLemmatizer()

stop_words = stopwords.words('english')
stopwords_dict = Counter(stop_words)

# Define stopwords dictionary
stop_words = stopwords.words('english')
stopwords_dict = set(stop_words)

# Load the TF-IDF vectorizer
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Preprocessing function
def nltk_preprocess(text):
    text = clean_text(text)
    wordlist = re.sub(r'[^\w\s]', '', text).split()
    text = ' '.join([wnl.lemmatize(word) for word in wordlist if word not in stopwords_dict])
    return  text

# Function to clean text
def clean_text(text):
    text = str(text).replace(r'http[\w:/\.]+', ' ')  # removing urls
    text = str(text).replace(r'[^\.\w\s]', ' ')  # remove everything but characters and punctuation
    text = str(text).replace('[^a-zA-Z]', ' ')
    text = str(text).replace(r'\s\s+', ' ')
    text = text.lower().strip()
    return text

# Prediction function
def predict_authenticity(text, model_name):
    cleaned_text = nltk_preprocess(text)
    tfidf_vector = vectorizer.transform([cleaned_text])
    model_path = f'best_{model_name}_model.pkl'
    model = joblib.load(model_path)
    prediction = model.predict(tfidf_vector)[0]
    return prediction

# Streamlit app
def main():
    st.title("News Authenticity Checker")
    st.write("Welcome to the News Authenticity Checker! This app helps you determine whether a piece of news is authentic or not.")

    # Instructions
    st.subheader("How to Use:")
    st.write("1. Enter a piece of news text in the text area below.")
    st.write("2. Choose a machine learning model from the dropdown menu.")
    st.write("3. Click on the 'Check Authenticity' button to see the prediction.")

    # Text input
    news_text = st.text_area("Enter the news text here:", "")

    # Model selection
    selected_model = st.selectbox("Choose a model:", ("AdaBoostClassifier()", "DecisionTreeClassifier()", "KNeighborsClassifier()", "LogisticRegression()", "MultinomialNB()", "NuSVC()", "RandomForestClassifier(n_jobs=-1)", "SVC()"))

    if st.button("Check Authenticity"):
        if news_text:
            # Make prediction
            prediction = predict_authenticity(news_text, selected_model.replace(" ", ""))
            if prediction == 1:
                st.success("The news is authentic.")
            else:
                st.error("The news is not authentic.")
        else:
            st.warning("Please enter some news text.")

    # Sample news articles
    st.sidebar.subheader("Sample News Articles:")
    sample_news = [
        "Scientists discover new planet orbiting distant star.",
        "Trump administration announces new trade tariffs on Chinese goods.",
        "Health experts warn about the spread of a new contagious disease.",
        "Local community celebrates opening of new park.",
        "Study finds link between coffee consumption and improved cognitive function."
    ]
    selected_sample = st.sidebar.selectbox("Choose a sample news article:", sample_news)
    st.sidebar.text_area("Selected sample news article:", selected_sample)

if __name__ == "__main__":
    main()
