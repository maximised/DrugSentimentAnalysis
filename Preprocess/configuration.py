# We will store all the configuration variables here
OVERALL_NEGATIVE_THRESHOLD = 2.5
OVERALL_POSITIVE_THRESHOLD = 4

VADER_NEGATIVE_THRESHOLD = -0.75
VADER_POSITIVE_THRESHOLD = -0.2

BERT_NEGATIVE_THRESHOLD = -0.79
BERT_POSITIVE_THRESHOLD = -0.2

#############################
# Drug Families
TCA = ['amitriptyline', 'clomipramine', 'coxepin', 'cortriptyline', 'imipramine', 'dosulepin']
MAOI = ['tranylcypromine', 'moclobemide', 'phenelzine', 'selegiline', 'isocarboxazid']
SSRI = ['fluoxetine', 'paroxetine', 'sertraline', 'citalopram', 'escitalopram', 'fluvoxamine']
SNRI = ['venlafaxine', 'duloxetine', 'desvenlafaxine', 'levomilnacipran', 'milnacipran']
Benzodiazepines = ['temazepam', 'nitrazepam', 'diazepam', 'oxazepam', 'alprazolam', 'lorazepam']
AtypicalAntipsychotics = ['aripiprazole', 'olanzapine', 'quetiapine', 'risperidone', 'ziprasidone', 'clozapine']
GABA = ['gabapentin', 'pregabalin', 'tiagabine', 'vigabatrin', 'valproate', 'carbamazepine']
MixedAntidepressants = ['bupropion', 'mirtazapine', 'trazodone']

Drugs = TCA + MAOI + SSRI + SNRI + Benzodiazepines + AtypicalAntipsychotics + GABA + MixedAntidepressants
DrugsDict = {}
for d in TCA:
    DrugsDict[d] = 'TCA'
for d in MAOI:
    DrugsDict[d] = 'MAOI'
for d in SSRI: 
    DrugsDict[d] = 'SSRI'
for d in SNRI:
    DrugsDict[d] = 'SNRI'
for d in Benzodiazepines:
    DrugsDict[d] = 'Benzodiazepines'
for d in AtypicalAntipsychotics:
    DrugsDict[d] = 'AtypicalAntipsychotics'
for d in GABA:
    DrugsDict[d] = 'GABA'
for d in MixedAntidepressants:
    DrugsDict[d] = 'MixedAntidepressants'

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
