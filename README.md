# Text Classification Project

This project implements a text classification system using three different algorithms: Naive Bayes, Linear SVM, and Logistic Regression. The goal is to classify input text into two categories: "it's not a disaster" or "it's a disaster."

## Table of Contents

- [Introduction](#introduction)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Gradio Interface](#gradio-interface)
- [Usage](#usage)
- [Requirements](#requirements)


## Introduction

In this project, we trained three different machine learning models on a dataset of tweets to classify their sentiment. The models used are:

1. **Naive Bayes**
2. **Linear Support Vector Machine (SVM)**
3. **Logistic Regression**

The models were evaluated based on their accuracy and performance on a test dataset.

## Data Preparation

The dataset used for training and testing is a collection of tweets labeled as either positive or negative sentiment. The data was preprocessed to remove unnecessary columns and clean the text.

### Data Cleaning Steps

- Removed HTML tags using BeautifulSoup.
- Converted text to lowercase.
- Removed special characters and numbers using regular expressions.
- Eliminated stopwords using NLTK.

## Model Training

The models were trained using three different pipelines:

1. **Naive Bayes Pipeline**
   - CountVectorizer
   - TfidfTransformer
   - MultinomialNB

2. **Linear SVM Pipeline**
   - CountVectorizer
   - TfidfTransformer
   - SGDClassifier (Linear SVM)

3. **Logistic Regression Pipeline**
   - CountVectorizer
   - TfidfTransformer
   - LogisticRegression

The models were saved as pickle files for later use.

## Gradio Interface

A Gradio user interface was created to allow users to classify text using the trained models. Users can input text and select one of the three algorithms to receive a classification result.

### Features of the Gradio Interface

- **Model Selection**: Users can choose between Naive Bayes, Linear SVM, and Logistic Regression.
- **Input Field**: Users can enter the text they wish to classify.
- **Output Display**: The interface shows whether the input text is classified as "it's not a disaster" or "it's a disaster."

## Usage

To use the application, run the Gradio interface, enter the text in the input box, select the desired model, and click the classify button. The output will indicate the classification result.

## Requirements

- Python 3.x
- Gradio
- Scikit-learn
- Joblib
- NLTK
- BeautifulSoup
- Pandas

