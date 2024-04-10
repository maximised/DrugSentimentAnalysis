# Description: This file contains the functions to predict the topic model of the reviews using the BERT model.
# The topic data are separated by Drug Family and sentiment, then saved in files

import pandas as pd
import numpy as np
import json
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import re

from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import remove_stopwords
import spacy
from gensim.models.phrases import Phrases, Phraser
from bertopic import BERTopic
from botocore.exceptions import NoCredentialsError

from configuration import *

nltk.download('stopwords')

nlp = spacy.load("en_core_web_md")

drug_families = list(set(DrugsDict.values()))
sentiments = ['negative','neutral','positive']

############################################################
# set an expanded df with reviews flattened
def expand_reviews(df):
    df = df.rename(columns={'EaseofUse': 'MeanEaseofUse', 'Effectiveness': 'MeanEffectiveness', 
                            'Satisfaction': 'MeanSatisfaction', 'Overall': 'MeanOverall'})
    expanded_df = df.explode('Reviews')
    expanded_df = pd.concat([expanded_df.drop(['Reviews'], axis=1), expanded_df['Reviews'].apply(pd.Series)], axis=1)
    return expanded_df

# Preprocess the text
def clean_text(text):
    text = text.lower()  # Lowercase text
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = remove_stopwords(text, stopwords=s_words)  # Remove stopwords
    return text

def lemmatize(text):
    return ' '.join([token.lemma_ for token in nlp(text)])

def create_bigram_phrases(docs):
    tokens = [doc.split() for doc in docs]
    phrases = Phrases(tokens, min_count=5, threshold=10)
    bigram = Phraser(phrases)
    return [' '.join(bigram[doc]) for doc in tokens]

def create_bigrams_and_trigrams(docs):
    # Tokenize the documents for phrase modeling
    tokens = [doc.split() for doc in docs]
    
    # Create bigrams
    bigram_phrases = Phrases(tokens, min_count=5, threshold=10)  # Tune these parameters as needed
    bigram = Phraser(bigram_phrases)
    bigram_tokens = [bigram[doc] for doc in tokens]
    
    # Use the bigram tokens to create trigrams
    trigram_phrases = Phrases(bigram_tokens, min_count=5, threshold=10)  # Tune these parameters as needed
    trigram = Phraser(trigram_phrases)
    trigram_tokens = [' '.join(trigram[doc]) for doc in bigram_tokens]
    
    return trigram_tokens

# Save models and data to files

from umap import UMAP

# Initialize CountVectorizer with English stopwords
vectorizer_model = CountVectorizer(
    ngram_range=(1, 3), 
    min_df=0.01, 
    max_df=0.6,
)

# Create a UMAP model instance with your parameters
umap_model = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, random_state=42)

#model_name = "dmis-lab/biobert-v1.1"
#sentence_model = SentenceTransformer(model_name)

def extract_topics_from_probabilities(topic_df, threshold = 0.05):
    # Assign documents to multiple topics based on the threshold
    multi_topics = [np.where(pd.Series(prob) >= threshold)[0] for prob in topic_df['TopicProbability']]

    topic_df['TopicsList'] = multi_topics
    
    return topic_df

# Step 2: Define a function to map topic numbers to names
def map_topics_to_names(topic_numbers, topic_info):
    topic_dict = topic_info.set_index('Topic')['Name'].to_dict()

    return [topic_dict[num] for num in topic_numbers if num in topic_dict]

def fit_bertopic(df, family):     
    # Initialize BERTopic
    if family != 'MAOI':
        topic_model = BERTopic(
            language="english", 
            calculate_probabilities=True, 
            #embedding_model="all-MiniLM-L6-v2",
            #min_topic_size=30,
            #n_gram_range=(1, 3),
            nr_topics='auto',
            umap_model=umap_model,
            vectorizer_model=vectorizer_model,
            #embedding_model=sentence_model
        )
    else:
        topic_model = BERTopic(
            language="english", 
            calculate_probabilities=True,
            umap_model=umap_model
        )

    # Fit the model to your documents
    topics, probabilities = topic_model.fit_transform(df['cleaned_Comment'].tolist())
    try:
        # Reduce outliers
        #topics = topic_model.reduce_outliers(df['cleaned_Comment'].tolist(), topics)
        #topics = topic_model.reduce_outliers(df['cleaned_Comment'].tolist(), topics,
        #                                     probabilities=probabilities,
        #                                     threshold=0.05, strategy="probabilities")
        topic_model.update_topics(df['cleaned_Comment'].tolist(), topics=topics)
        topic_model.reduce_topics(df['cleaned_Comment'].tolist(), 
                                nr_topics='auto')
    except Exception as e:
        print(e)

    df['Topic'] = topic_model.topics_
    df['TopicName'] = df['Topic'].map(topic_model.topic_labels_)
    df['TopicProbability'] = topic_model.probabilities_.tolist()
    # Get list of most likely topics
    df = extract_topics_from_probabilities(df, threshold = 0.05)
    df['TopicNames'] = df['TopicsList'].apply(lambda l: map_topics_to_names(l, topic_model.get_topic_info()))
    # Get UMAP embeddings
    df[['UMAP_x','UMAP_y']] = topic_model.umap_model.embedding_[:, :2]

    return df, topic_model

########################################

bertopic_out = {}
bertopic_path = '../data/bertopic'
s3_bertopic_path = 'data/bertopic'

def save_bertopic(drug_families, sentiments):
    for family in drug_families:
        for sentiment in sentiments:
            print(family, sentiment)

            bertopic_df = expanded_df[(expanded_df['DrugFamily'] == family) & (expanded_df['bert_compound_threshold'] == sentiment)]
            bertopic_df, topic_model = fit_bertopic(bertopic_df, family)
            
            path = bertopic_path+ '/' +family+'_'+sentiment
            os.makedirs(path, exist_ok=True)
            # Example: Saving a DataFrame
            #bertopic_df.to_csv(path+ '/' + 'df.csv', index=False)
            bertopic_df.to_json(path+ '/' + 'df.csv', orient='records', lines=True)
            # For spaCy models (as an example of sentiment or topic models)
            topic_model.save(path+ '/' + 'model')

            # Serialize the results dictionary to JSON
            bertopic_out[family + '_' + sentiment] = {'family': family, 'sentiment': sentiment, 'df': path+'/'+'df.csv', 'model': path+'/'+'model'}
    with open(bertopic_path+ '/' +'json_out.json', 'w') as f:
        json.dump(bertopic_out, f, indent=4)


# Initialize boto3 client
def save_bertopic_to_s3(drug_families, sentiments, bucket='maxim-thesis'):
    s3_client = boto3.client('s3', region_name='eu-west-1')
    
    def upload_file_to_s3(file_name, bucket, s3_file_name):
        try:
            s3_client.upload_file(file_name, bucket, s3_file_name)
            print("Upload Successful")
        except FileNotFoundError:
            print("The file was not found")
        except NoCredentialsError:
            print("Credentials not available")
        
    for family in drug_families:
        for sentiment in sentiments:
            print(family, sentiment)

            # Similar to your existing process, but add the step to upload to S3
            bertopic_df = expanded_df[(expanded_df['DrugFamily'] == family) & (expanded_df['bert_compound_threshold'] == sentiment)]
            bertopic_df, topic_model = fit_bertopic(bertopic_df, family)
            
            path = bertopic_path+ '/' +family+'_'+sentiment
            s3_path = s3_bertopic_path+ '/' +family+'_'+sentiment
            
            os.makedirs(path, exist_ok=True)
            # Example: Saving a DataFrame
            bertopic_df.to_json(path+ '/' + 'df.csv', orient='records', lines=True)
            topic_model.save(path+ '/' + 'model')

            upload_file_to_s3(path+ '/' + 'df.csv', bucket, s3_path+ '/' + 'df.csv')
            upload_file_to_s3(path+ '/' + 'model', bucket, s3_path+ '/' + 'model')

            # Serialize the results dictionary to JSON
            bertopic_out[family + '_' + sentiment] = {'family': family, 'sentiment': sentiment, 'df': s3_path+'/'+'df.csv', 'model': s3_path+'/'+'model'}
    
    with open(bertopic_path+ '/' +'json_out.json', 'w') as f:
        json.dump(bertopic_out, f, indent=4)
    # save bertopic_out to s3
    upload_file_to_s3(bertopic_path+ '/' +'json_out.json', bucket, s3_bertopic_path+ '/' +'json_out.json')


############################################################
# Load the dataset
#df = pd.read_json('../data/2_webmd_preprocessed_sentiment.csv', orient='records', lines=True)
print('reading from s3')
df = load_from_s3('maxim-thesis', 'data/2_webmd_preprocessed_sentiment.csv')
# Expand the df into single reviews
expanded_df = expand_reviews(df)

# Set the custom stop words from the drugs df
custom_stop_words = ['drug', 'medication', 'taking','take','feel','would',
                     'could','not','make','get','go','mg','like']
drug_names = list(set(expanded_df['Drug'])) + \
    ['prozac','paxil']
s_words = stopwords.words('english') \
    + custom_stop_words \
    + drug_names

# Preprocess the text
print('Preprocessing the text')
expanded_df['cleaned_Comment'] = expanded_df['Comment']\
    .apply(clean_text)\
    .apply(lemmatize)\
    .apply(clean_text)
expanded_df['cleaned_Comment_with_phrases'] = create_bigrams_and_trigrams(expanded_df['cleaned_Comment'])

# Apply bertopic and save the results and models in files
print('Applying BERTopic')
#save_bertopic(drug_families, sentiments)
save_bertopic_to_s3(drug_families, sentiments)