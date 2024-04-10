import plotly.express as px
from bertopic import BERTopic
import json

colors = {
    'background': '#FFFFFF',
    'text': '#7FDBFF'
}
col_swatch = px.colors.qualitative.Dark24

# The rating categories
ratings = ['Effectiveness', 'EaseOfUse', 'Satisfaction', 'Overall']

# The columns to display for the individual reviews
review_cols = ['BrandName','DrugCondition','Drug','GenName','DrugFamily','Condition','IsPatient',
               'IsMale','Age','TimeUsingDrug','Topic','TopicName']

# Assuming you have a list of options for your dropdown
sentiments = ['positive', 'negative', 'neutral']

# Model visualised
MODEL = 'bert'

time_using_drug_order = ['less than 1 month', '1 to 6 months', '6 months to less than 1 year', 
                             '1 to less than 2 years', '2 to less than 5 years', '5 to less than 10 years', '10 years or more']

#############################
# S3 Functions
import boto3
import pandas as pd
from io import StringIO

def save_to_s3(df, bucket_name, file_key):
    # Save a df to S3
    json_str = df.to_json(orient='records', lines=True)

    # Convert the JSON string to bytes
    json_bytes = json_str.encode('utf-8')

    s3 = boto3.resource('s3', region_name='eu-west-1')

    s3_object = s3.Object(bucket_name, file_key)
    s3_object.put(Body=json_bytes)

    return

def load_from_s3(bucket_name, file_key):
    s3 = boto3.client('s3', region_name='eu-west-1')

    json_obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    json_file_body = json_obj['Body'].read()
    json_str = json_file_body.decode('utf-8')

    df = pd.read_json(StringIO(json_str), orient='records', lines=True)

    return df

import tempfile
from io import BytesIO

s3_resource = boto3.resource('s3', region_name='eu-west-1')
def read_bertopic_from_s3(family, sentiment, bucket_name='maxim-thesis'):
    name = family + '_' + sentiment
    json_key = f'data/bertopic/json_out.json'

    # Load the JSON output
    json_object = s3_resource.Object(bucket_name, json_key)
    json_file = json_object.get()['Body'].read().decode('utf-8')
    loaded_results = json.loads(json_file)

    # Load the DataFrame from S3
    df_object = s3_resource.Object(bucket_name, loaded_results[name]['df'])
    df_buffer = BytesIO(df_object.get()['Body'].read())
    topic_df = pd.read_json(df_buffer, orient='records', lines=True)

    # Load the BERTopic model from S3
    model_object = s3_resource.Object(bucket_name, loaded_results[name]['model'])
    model_buffer = model_object.get()['Body'].read()
    # Create a temporary file to save the model
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file_name = tmp_file.name
        tmp_file.write(model_buffer)
        tmp_file.flush()
    # Load the BERTopic model from the temporary file
    topic_model = BERTopic.load(tmp_file_name)

    return family, sentiment, topic_df, topic_model