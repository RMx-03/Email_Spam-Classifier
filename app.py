# Necessary libraries for the app
import streamlit as st
import pickle
import numpy as np
import joblib  


# Load the trained model
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')


# Streamlit UI
st.title("Email Spam Classifier")
st.write("Enter an email message below to classify as **Ham** or **Spam**.")


# Text area for user input
email_text = st.text_area("Enter your email text here:", height=200)


# Button to classify the email
# Making prediction and displaying the result
if st.button("Classify Email"):
    if email_text.strip() == "":
        st.warning("Please enter an email message.")
    else: 
        # preprocess and transform the input
        email_features = vectorizer.transform([email_text]).toarray()

        # making prediction
        prediction = model.predict(email_features)[0]

        # display the prediction
        if prediction == 0:
            st.error("This email is **Spam**.")
        else:
            st.success("This email is **Ham**.")
     

st.write("**Model Used**: Bernoulli Naive Bayes")