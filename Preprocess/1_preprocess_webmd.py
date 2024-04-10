# This file preprocesses the webmd dataset

import pandas as pd
import numpy as np
import ast
from configuration import *

###################################
# Replace empty strings with NaN and change to numeric type
def preprocess_num_cols(df):
    # Convert the 'value_to_change' from string to integer
    # Calculate the overall rating
    ratings = ['Effectiveness', 'EaseofUse', 'Satisfaction']
    f_cols = ['EaseofUse', 'Effectiveness', 'Satisfaction']
    i_cols = ['NumReviews']
    
    df[f_cols] = df[f_cols].replace(' ', np.nan).astype(float)
    df[i_cols] = df[i_cols].replace(' ', np.nan).astype(int)

    df['Overall'] = df[ratings].mean(axis=1)

    return df

###################################
# Function to modify the dictionary items 

def modify_dict_bool(l):
    # Convert the 'value_to_change' from string to integer
    mapping = {'True': True, 'False': False, ' ': np.nan}
    cols = ['IsMale', 'IsPatient']
    
    try:
        for d in l:
            for col in cols:
                d[col] = mapping[d[col]]
    except KeyError:
        pass

    return l

def modify_dict_int(l):
    # Convert the 'value_to_change' from string to integer
    cols = ['Effectiveness', 'EaseOfUse','Satisfaction','NumFoundHelpful','NumVoted']
    mapping = [' ', None]

    try:
        for d in l:
            for col in cols:
                # convert to float
                d[col] = np.nan if d[col] in mapping else d[col]
                d[col] = float(d[col])

            # Overall rating is average of other ratings
            d['Overall'] = np.mean([ d['Effectiveness'], d['EaseOfUse'], d['Satisfaction'] ])
    except KeyError:
        pass

    return l

def modify_dict_date(l):
    # Convert the 'value_to_change' from string to integer
    cols = ['DatePosted']
    date_format = '%m/%d/%Y %I:%M:%S %p'
    mapping = [' ', None]

    try:
        for d in l:
            for col in cols:
                d[col] = pd.NaT if d[col] in mapping else d[col]
                d[col] = pd.to_datetime(d[col], format=date_format, errors='coerce').to_pydatetime()
    except KeyError:
        pass

    return l


###################################
# Functions to filter drugs

# Function to map complex names to simpler names
def simplify_drug_name(name):
    for d in DrugsDict.keys():
        if d in name:
            return d
    return name

def update_genname_with_drug(row):
    if row['GenName'] == ' ':
        for d in DrugsDict.keys():
            if d in row['Drug']:
                row['GenName'] = row['Drug']
                return row
    return row

def update_genname(df):
    df = df.apply(lambda row: update_genname_with_drug(row), axis=1)
    df['GenName'] = df['GenName'].apply(lambda x: simplify_drug_name(x))
    return df

def filter_drugs(df, DrugsDict):
    # Update and simply GenName column
    df = update_genname(df)

    # Filter out drugs that GenName or Drug Name not in DrugsDict
    #df = df[df['GenName'].apply(lambda x: any(d in x for d in DrugsDict.keys()))]
    df = df[df.apply(lambda r: 
                     any(d in r['GenName'] for d in DrugsDict.keys()) | 
                     any(d in r['Drug'] for d in DrugsDict.keys())
                    , axis=1)]
    df['DrugFamily'] = df['GenName'].map(DrugsDict) # Find Family of Drug
    df = df.drop_duplicates(subset=['DrugId', 'GenName'], keep='first')
    df = df[df['DrugFamily'].notna()]
    
    return df

##########################################
def filter_reviews_with_no_comment(dict_list):
    # Filter out reviews with no comment
    filtered_list = [d for d in dict_list if len(d['Comment']) > 2 ]
    return filtered_list

def mean_of_key(dict_list, key):
    # Extract values associated with the key, ignoring missing keys and NaN values
    values = [d[key] for d in dict_list if key in d and pd.notna(d[key])]
    # Calculate and return the mean, return NaN if values list is empty
    if values:
        return np.mean(values)
    else:
        return np.nan

##########################################
def map_ages(df):
    age_mapping = {'0-2': '00-02', '3-6': '03-06', '7-12': '07-12'}
    #df['Reviews'] = df['Reviews'].map(lambda x: [{'Age': age_mapping.get(review['Age'], review['Age'])} for review in x])
    for row in df['Reviews']:
        for review in row:
            review['Age'] = age_mapping.get(review['Age'], review['Age'])
    return df

def add_yrs_to_age(df):
    # This is needed because some plots fuck up otherwise
    for row in df['Reviews']:
        for review in row:
            review['Age'] = review['Age'] + ' yrs'
    return df

###################################
# Read in the dataset
#full_df = pd.read_csv('../data/0_webmd.csv')
print('reading from s3')
full_df = load_from_s3('maxim-thesis', 'data/0_webmd.csv')
print('Preprocessing webmd dataset')

# Filter to drugs only
df = filter_drugs(full_df, DrugsDict)
# Change Review column to list of dictionaries
df['Reviews'] = df['Reviews'].apply(lambda d: ast.literal_eval(d))
# Preprocess the numerical columns
df = preprocess_num_cols(df)
# Preprocess the categorical columns in Reviews
df['Reviews'] = df['Reviews'].apply(modify_dict_bool)
# Preprocess the numerical columns in Reviews
df['Reviews'] = df['Reviews'].apply(modify_dict_int)
# Prepocess the date columns in Reviews
df['Reviews'] = df['Reviews'].apply(modify_dict_date)
# Remove na rows
df = df.dropna()
# Remove any reviews with no comments
df['Reviews'] = df['Reviews'].apply(filter_reviews_with_no_comment)
# Add a column for the number of reviews
df['ScrapedReviews'] = df['Reviews'].apply(lambda l: len(l))
# Remove any drugs with no reviews
df = df[df['ScrapedReviews'] > 0]
# Get mean of the ratings (from remaining reviews)
df['MeanEffectiveness'] = df['Reviews'].apply(lambda l: mean_of_key(l, 'Effectiveness'))
df['MeanEaseOfUse'] = df['Reviews'].apply(lambda l: mean_of_key(l, 'EaseOfUse'))
df['MeanSatisfaction'] = df['Reviews'].apply(lambda l: mean_of_key(l, 'Satisfaction'))
df['MeanOverall'] = df['Reviews'].apply(lambda l: mean_of_key(l, 'Overall'))
df = df.drop(columns=['Effectiveness','EaseofUse','Satisfaction','Overall'])
# Rename the columns that are in Reviews column
df = df.rename(columns={'Condition': 'DrugCondition'})
# Map ages
df = map_ages(df)
df = add_yrs_to_age(df)

# Save to JSON with orient='records' to keep the list of dictionaries intact
print('saving to s3')
#df.to_json('../data/1_webmd_preprocessed.csv', orient='records', lines=True)
save_to_s3(df, 'maxim-thesis', 'data/1_webmd_preprocessed.csv')