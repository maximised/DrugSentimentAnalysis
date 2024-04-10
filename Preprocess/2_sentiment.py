# Description: This file contains the functions to predict the sentiment of the reviews using the VADER model and RoBERTa model.

import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import joblib

from configuration import *


############################################################
# Fit the VADER model to the dataset
def vader_sentiment_df(df, column = 'Reviews'):
    # Predict emotions in the review column

    def vader_sentiment(sentence, sid_obj = SentimentIntensityAnalyzer()):
        sentiment_dict = sid_obj.polarity_scores(sentence)
        return sentiment_dict

    def predict_emotion(list_reviews):
        for i in range(len(list_reviews)):
            sentiment = vader_sentiment(list_reviews[i]['Comment'])
            list_reviews[i]['vader_neg'] = sentiment['neg']
            list_reviews[i]['vader_neu'] = sentiment['neu']
            list_reviews[i]['vader_pos'] = sentiment['pos']
            list_reviews[i]['vader_compound'] = sentiment['compound']
        return list_reviews

    df['Reviews'] = df['Reviews'].apply(lambda l: predict_emotion(l))
    return df

############################################################
def bert_sentiment_df(df, column = 'Reviews', model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Load the model; since it's not specifically a sentiment analysis model, we use the base model
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def bert_sentiment(text, tokenizer, model):
        # encode text
        encoded_text = tokenizer(text, return_tensors="pt")
        # Get model predictions
        output = model(**encoded_text)

        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        return scores
    
    # Predict emotions in the review column
    def predict_emotion(list_reviews):
        for i in range(len(list_reviews)):
            try:
                sentiment = bert_sentiment(list_reviews[i]['Comment'], tokenizer, model)
            except:
                print('Error in:', list_reviews[i]['Comment'])
                sentiment = [np.nan,np.nan,np.nan]
            list_reviews[i]['bert_neg'] = sentiment[0]
            list_reviews[i]['bert_neu'] = sentiment[1]
            list_reviews[i]['bert_pos'] = sentiment[2]
            list_reviews[i]['bert_compound'] = sentiment[2] - sentiment[0]
        return list_reviews
    
    df['Reviews'] = df['Reviews'].apply(lambda l: predict_emotion(l))
    return df
'''
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from scipy.special import softmax
from concurrent.futures import ProcessPoolExecutor, as_completed

# Assuming the model and tokenizer are loaded outside the function to avoid reloading them multiple times
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model.eval() # Set the model to evaluation mode

if torch.cuda.is_available():
    model.to('cuda')

def bert_sentiment(text, tokenizer, model):
    with torch.no_grad(): # Disable gradient calculation
        encoded_text = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        if torch.cuda.is_available():
            encoded_text = encoded_text.to('cuda')
        output = model(**encoded_text)
        scores = output.logits[0].cpu().numpy() # Move logits to CPU
        scores = softmax(scores)
    return scores

def predict_emotion(review, tokenizer, model):
    try:
        sentiment = bert_sentiment(review['Comment'], tokenizer, model)
    except Exception as e:
        print('Error in:', review['Comment'], "; Error:", str(e))
        sentiment = [np.nan, np.nan, np.nan]
    review['bert_neg'] = sentiment[0]
    review['bert_neu'] = sentiment[1]
    review['bert_pos'] = sentiment[2]
    review['bert_compound'] = sentiment[2] - sentiment[0]
    return review

def bert_sentiment_df(df, column='Reviews', model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
    # Assume df['Reviews'] is a list of dictionaries with 'Comment' key
    def process_reviews(reviews):
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(predict_emotion, review, tokenizer, model) for review in reviews]
            results = []
            for future in as_completed(futures):
                results.append(future.result())
        return results
    
    df[column] = df[column].apply(process_reviews)
    return df'''

# Usage: Ensure df['Reviews'] is formatted correctly
# optimized_df = bert_sentiment_df(your_dataframe)


###############################################
# Categorise the Overall, VADER, BERT scores to negative, neutral or negative
# Set thresholds for Overall Rating

def apply_thresholf_df(df, score, NEGATIVE_THRESHOLD, POSITIVE_THRESHOLD):

    def apply_threshold(score, NEGATIVE_THRESHOLD, POSITIVE_THRESHOLD):
        if score <= NEGATIVE_THRESHOLD:
            return 'negative'
        elif score >= POSITIVE_THRESHOLD:
            return 'positive'
        else:
            return 'neutral'
    
    # Predict emotions in the review column
    def apply_threshold_series(list_reviews):
        for i in range(len(list_reviews)):
            try:
                threshold = apply_threshold(list_reviews[i][score], NEGATIVE_THRESHOLD, POSITIVE_THRESHOLD)
            except:
                print('Error in:', list_reviews[i][score])
                threshold = 'neutral'
            list_reviews[i][score+'_threshold'] = threshold
        return list_reviews
    
    df['Reviews'] = df['Reviews'].apply(lambda l: apply_threshold_series(l))
    return df

##########################################
def filter_reviews_with_nan_bert_sentiment(dict_list):
    # Filter out reviews with no comment
    filtered_list = [d for d in dict_list if (d['bert_compound'] is not None) ]
    filtered_list = [d for d in filtered_list if (-1 <= d['bert_compound'] <= 1) ]
    return filtered_list


def mean_of_key(dict_list, key):
    # Extract values associated with the key, ignoring missing keys and NaN values
    values = [d[key] for d in dict_list if key in d and pd.notna(d[key])]
    # Calculate and return the mean, return NaN if values list is empty
    if values:
        return np.mean(values)
    else:
        return np.nan
    
############################################################
# set an expanded df with reviews flattened
def expand_reviews(df):
    df = df.rename(columns={'EaseofUse': 'MeanEaseofUse', 'Effectiveness': 'MeanEffectiveness', 
                            'Satisfaction': 'MeanSatisfaction', 'Overall': 'MeanOverall'})
    expanded_df = df.explode('Reviews')
    expanded_df = pd.concat([expanded_df.drop(['Reviews'], axis=1), expanded_df['Reviews'].apply(pd.Series)], axis=1)
    return expanded_df

############################################################
# Get Rating from the BERT model
def create_sentiment_to_rating_model(sentiment_model, expanded_df):

    plot_df = expanded_df[(expanded_df['vader_compound'].notna()) & (expanded_df['bert_compound'].notna())]

    # Create synthetic data for demonstration (replace with your actual data)
    sentiment_scores = plot_df[sentiment_model+'_compound'].values
    star_ratings = plot_df['Overall'].values

    # Reshape data for the model
    X = sentiment_scores.reshape(-1, 1)
    y = star_ratings.reshape(-1, 1)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit a polynomial regression model
    degree = 5  # Degree of polynomial. You can adjust this based on your validation.
    polynomial_features = PolynomialFeatures(degree=degree)
    linear_regression = LinearRegression()

    # Using a pipeline to combine steps
    model = make_pipeline(polynomial_features, linear_regression)
    model.fit(X_train, y_train)

    #joblib.dump(model, '../data/sentiment/model.pkl')
    print('Saved model to data/sentiment/model.pkl')

    return model

def predict_sentiment_to_rating(l, model):
    # Use this to alter the df with the dictionary items
    try:
        for d in l:
            d['PredictedOverall'] = model.predict([[d['bert_compound']]])[0][0]
    except KeyError:
        pass

    return l

############################################################
# Load the dataset
#df = pd.read_json('../data/1_webmd_preprocessed.csv', orient='records', lines=True)
print('reading from s3')
df = load_from_s3('maxim-thesis', 'data/1_webmd_preprocessed.csv')

# Fit the VADER model to the dataset
print('Fitting VADER model to the dataset')
df = vader_sentiment_df(df)
# Fit the RoBERTa model to the dataset
print('Fitting RoBERTa model to the dataset')
df = bert_sentiment_df(df)

# Remove any reviews with nan bert sentiment
df['Reviews'] = df['Reviews'].apply(filter_reviews_with_nan_bert_sentiment)
# Add a column for the number of reviews
df['ScrapedReviews'] = df['Reviews'].apply(lambda l: len(l))
# Remove any drugs with no reviews
df = df[df['ScrapedReviews'] > 0]
# Get mean of the ratings (from remaining reviews)
df['MeanEffectiveness'] = df['Reviews'].apply(lambda l: mean_of_key(l, 'Effectiveness'))
df['MeanEaseOfUse'] = df['Reviews'].apply(lambda l: mean_of_key(l, 'EaseOfUse'))
df['MeanSatisfaction'] = df['Reviews'].apply(lambda l: mean_of_key(l, 'Satisfaction'))
df['MeanOverall'] = df['Reviews'].apply(lambda l: mean_of_key(l, 'Overall'))

# Categorise the Overall, VADER, BERT scores to negative, neutral or negative
df = apply_thresholf_df(df, 'Overall', OVERALL_NEGATIVE_THRESHOLD, OVERALL_POSITIVE_THRESHOLD)
df = apply_thresholf_df(df, 'vader_compound', VADER_NEGATIVE_THRESHOLD, VADER_POSITIVE_THRESHOLD)
df = apply_thresholf_df(df, 'bert_compound', BERT_NEGATIVE_THRESHOLD, BERT_POSITIVE_THRESHOLD)

# Calculate the Predicted Overall score from bert_compound
expanded_df = expand_reviews(df)
model = create_sentiment_to_rating_model('bert', expanded_df)
df['Reviews'] = df['Reviews'].apply(lambda l: predict_sentiment_to_rating(l, model))

# Save to JSON with orient='records' to keep the list of dictionaries intact
print('Saving to JSON')
#df.to_json('../data/2_webmd_preprocessed_sentiment.csv', orient='records', lines=True)
save_to_s3(df, 'maxim-thesis', 'data/2_webmd_preprocessed_sentiment.csv')