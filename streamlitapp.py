import streamlit as st 
import joblib,os
import spacy
import pandas as pd

# Load the spacy model
nlp = spacy.load("en_core_web_sm")

# Import matplotlib
import matplotlib
import matplotlib.pyplot as plt 

# Use matplotlib for rendering
matplotlib.use("Agg")

# Import Image and WordCloud libraries from PIL
# from PIL import Image
# from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# Load the vectorization object from disk
news_vectorizer = open("vectorization.pkl","rb")
news_cv = joblib.load(news_vectorizer)


def main():
    # Display the header
    st.markdown("<h1 style='text-align: center; color: White;'>Fake News Detection</h1>", unsafe_allow_html=True)

    # Sidebar activity selector
    activities = ['Prediction']
    choice = st.sidebar.selectbox('Choose Activity', activities)

    # Prediction section
    if choice == 'Prediction':
        st.info('Prediction with ML')
        # Text input field
        news_text = st.text_area('Enter Text')
        # Model selector
        ml_model = ['Decision Tree Classifier']
        model_choice = st.selectbox('Choose ML Model', ml_model)
        prediction_labels = {'Fake' : 0, 'Real': 1}
        # Predict button
        if st.button("Classify"):
            st.text("Original Text::\n{}".format(news_text))
            # Transform the input text
            vect_text = news_cv.transform([news_text]).toarray()
            # Load the selected model from disk
            if model_choice == 'Decision Tree Classifier':
                predictor = joblib.load("Decision_TC_model.pkl")
                
            # Make the prediction
            prediction = predictor.predict(vect_text)
            prediction_label = prediction[0]
            if prediction_label == 0:
                st.warning("Fake News")
            else:
                st.success("Real News")

# Check if the script is being run as the main program
if __name__ == '__main__':
    main()

