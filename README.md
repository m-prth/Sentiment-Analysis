# Sentiment Analysis on IMDB Movie Reviews
> Made by Parth Mistry

* Contains 50K movie reviews for natural language processing or Text analytics.
* This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets.
* We have a set of 25,000 highly polar movie reviews for training and 25,000 for testing. 
* So, predict the number of positive and negative reviews using either classification or deep learning algorithms.
* Here we will be using Logistic Regression to classify the reviews.

## Code and Resources
**Python Version:** 3.7      
**Packages:** pandas, numpy, sklearn, nltk, pickle   
**Dataset:** [IMDB Movie Reviews](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/download)

## Steps Performed
1. Transforming Documents to Feature Vectors
2. Checking word relevancy using TF-IDF
3. Calculating TF-IDF of each term
4. Removing noisy data
5. Tokenization of documents
6. Transforming Text Data into TF-IDF Vectors
7. Document Classsification using Logistic regression

## Model Preparation
```  
LogisticRegressionCV(cv=5,
                    scoring='accuracy',
                    random_state=0,
                    n_jobs=-1,
                    verbose=3,
                    max_iter=300)
```

* Here, I used Logistic Regression on the cleaned data, and it was trained with 89% of accuracy classifying movie reviews.
Parth Mistry © 2020











