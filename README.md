# Sentiment Analysis using NLP and Deep Learning

## Overview

This project implements a deep learning-based sentiment analysis model with real-time prediction capabilities. It classifies text into different sentiment categories with high accuracy, leveraging Natural Language Processing (NLP) techniques and deep learning architectures.

## Key Features

Achieved 96.58% accuracy in sentiment classification.

Utilized NLP techniques for text preprocessing and feature extraction.

Supports real-time sentiment prediction for dynamic analysis.

# Installation

Clone the repository:
```sh
git clone https://github.com/utkarsh-2706/Sentiment-Analysis-using-NLP.git
cd Sentiment-Analysis-using-NLP
```
Install dependencies:
```sh
pip install -r requirements.txt
```
# Usage

1. Use the pre-trained model from the repository:
```sh
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

# Load the dataset word index
word_index = imdb.get_word_index()
reversed_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model
model = load_model('simplernn_imdb.h5')
model.summary()

# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reversed_word_index.get(i-3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Function to predict sentiment
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

# Example usage
example_review = "this product is really comfortable and easy to use."
sentiment, score = predict_sentiment(example_review)

print(f"This review was", sentiment)
print(f"Score", score)
```
2. Alternatively, create your own model using the steps provided in the Jupyter notebook and save it as sentiment_model.h5.
