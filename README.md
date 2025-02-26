# Email Spam Classifier using Naive Bayes 📧

This project implements an email spam classification system using Naïve Bayes classifiers. The model is trained on preprocessed email text data and deployed using Streamlit for real-time spam detection.

---

## Table of Content 📜
- **Data**: Gathering and loading Dataset as well as Cleaning.
- **Exploratory Data Analysis**: To understand the gathered data. 
- **Preprocessing**: Stop Word removal, Stemming/Lemmatization, Tokenization using NLTK library, feature extraction using TF-IDF Vectorizer.
- **Model Training**: Implemented GaussianNB, BernoulliNB, and MultinomialNB classifiers.
- **Model Evaluation**: (Performance Metrics) Evaluated model using accuracy score, confusion matrix, and precision score.
- **Model Improvisation**: (Hyperparameter Tuning) Used GridSearchCV for optimizing BernoulliNB.
- **Deployment**: Built a Streamlit web app for real-time email classification.

- **Accuracy**: ~97.85% 

---

## How to Run 🖥️

To deploy this project run:

```bash
    streamlit run app.py
```