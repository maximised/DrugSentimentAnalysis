# Import required libraries
import dash
from dash import html, dcc, Input, Output, dash_table
#import dash_design_kit as ddk
import plotly.express as px
import pandas as pd
import numpy as np
import json
from bertopic import BERTopic
from scipy.stats import pointbiserialr, spearmanr
import dash_bootstrap_components as dbc
import datetime
import dash_bootstrap_components as dbc
import boto3
from io import BytesIO
import tempfile

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ipywidgets import interact, Dropdown, IntSlider

from DrugsDict import *
from configuration import *

########################################
# set an expanded df with reviews flattened
def expand_reviews(df):
    df = df.rename(columns={'EaseofUse': 'MeanEaseOfUse', 'Effectiveness': 'MeanEffectiveness', 
                            'Satisfaction': 'MeanSatisfaction', 'Overall': 'MeanOverall'})
    expanded_df = df.explode('Reviews')
    expanded_df = pd.concat([expanded_df.drop(['Reviews'], axis=1), expanded_df['Reviews'].apply(pd.Series)], axis=1)
    return expanded_df

########################################
# Group the data by DrugFamily and calculate the mean of each rating
def plot_ratings_by_family(df):
    # Group the data by DrugFamily and calculate the mean of each rating
    grouped = df.groupby(['DrugFamily'], as_index=0, dropna=0).agg({'EaseOfUse': 'mean', 'Effectiveness': 'mean', 'Satisfaction': 'mean', 'Overall': 'mean'})
    grouped = grouped.sort_values(['Overall'], ascending=False)
    
    # Initialize a figure
    fig = go.Figure()

    # Add bar plots for each rating type
    for rating in ['Effectiveness', 'EaseOfUse', 'Satisfaction']:
        fig.add_trace(go.Bar(
            x=grouped['DrugFamily'],
            y=grouped[rating],
            name=rating
        ))

    # Adjusting the layout for better visualization
    fig.update_layout(
        barmode='group',
        title='Ratings by Drug Family',
        xaxis_title='Drug Family',
        yaxis_title='Ratings',
        legend_title='Rating Type'
    )

    # Add horizontal lines for Overall values
    c = 0
    for index, row in grouped.iterrows():
        fig.add_shape(type='line',
                    x0=-0.5 + c,
                    y0=row['Overall'],
                    x1=0.5 + c,
                    y1=row['Overall'],
                    line=dict(color='black'),
                    xref='x',
                    yref='y')
        c+=1

    # Add Overall values as annotations
    for i, row in grouped.iterrows():
        fig.add_annotation(x=row['DrugFamily'], y=4.5,  # Adjust the offset as needed
                        text=round(row['Overall'],2),
                        showarrow=False,
                        yshift=10)  # Adjust the shift as needed for better positioning

    # Show the plot
    fig.update_layout(yaxis_range=[0,5])

    return fig

def rating_distribution(df):
    # Create a figure with 2 rows and 3 columns for 4 subplots
    # Specify the first plot to span all three columns in the first row
    fig = make_subplots(rows=2, cols=3, 
                        subplot_titles=('Overall', 'Effectiveness', 'EaseOfUse', 'Satisfaction'),
                        specs=[[{"colspan": 3}, None, None],
                               [{}, {}, {}]])

    # Plot the distribution of Overall ratings in the first row
    frequencies = df['Overall'].value_counts(normalize=True).reset_index().sort_values('Overall')
    frequencies.columns = ['Overall', 'Frequency']
    fig.add_trace(go.Bar(x=frequencies['Overall'], y=frequencies['Frequency'], name='Overall'), row=1, col=1)

    # Plot the distributions of individual ratings in the second row
    cols = ['Effectiveness', 'EaseOfUse', 'Satisfaction']
    for i, col in enumerate(cols, start=1): # Place these distributions in the second row
        frequencies = df[col].value_counts(normalize=True).reset_index().sort_values(col)
        frequencies.columns = [col, 'Frequency']
        fig.add_trace(go.Bar(x=frequencies[col], y=frequencies['Frequency'], name=col), row=2, col=i)

    # Update layout
    fig.update_layout(
        #height=400, width=600, 
        title_text="Distribution of Ratings by Category", showlegend=False
    )
    
    # Show the plot
    return fig

########################################
# Assuming 'expanded_df' is your DataFrame and is already loaded
def plot_demographics(expanded_df):
    # Custom order for age groups
    age_group_order = [" ", "0-2", "3-6", "7-12", "13-18", "19-24", "25-34", 
                    "35-44", "45-54", "55-64", "65-74", "75 or over"]

    # Create subplots
    fig = make_subplots(rows=1, cols=3, subplot_titles=("DrugFamily by IsMale", "DrugFamily by IsPatient", "Age Distribution"))

    # First plot: DrugFamily by IsMale
    df_male = expanded_df.groupby(['DrugFamily', 'IsMale']).size().reset_index(name='counts')
    for is_male in expanded_df['IsMale'].dropna().unique():
        fig.add_trace(go.Bar(x=df_male[df_male['IsMale'] == is_male]['DrugFamily'], 
                            y=df_male[df_male['IsMale'] == is_male]['counts'], 
                            name=str(is_male)), 
                    row=1, col=1)

    # Second plot: DrugFamily by IsPatient
    df_patient = expanded_df.groupby(['DrugFamily', 'IsPatient']).size().reset_index(name='counts')
    for is_patient in expanded_df['IsPatient'].dropna().unique():
        fig.add_trace(go.Bar(x=df_patient[df_patient['IsPatient'] == is_patient]['DrugFamily'], 
                            y=df_patient[df_patient['IsPatient'] == is_patient]['counts'], 
                            name=str(is_patient)), 
                    row=1, col=2)

    # Third plot: Age Distribution
    # Reorder the DataFrame based on your custom age group order
    expanded_df['Age'] = pd.Categorical(expanded_df['Age'], categories=age_group_order, ordered=True)
    df_age = expanded_df['Age'].value_counts().sort_index().reset_index()
    df_age.columns = ['Age', 'counts']

    fig.add_trace(go.Bar(x=df_age['Age'], y=df_age['counts']), row=1, col=3)

    # Update xaxis properties
    fig.update_xaxes(tickangle=-90, row=1, col=1)
    fig.update_xaxes(tickangle=-90, row=1, col=2)
    fig.update_xaxes(tickangle=-90, row=1, col=3)

    # Update layout for a better look
    fig.update_layout(
        #height=600, width=1800, 
        showlegend=True, title_text="Overview by DrugFamily, Patient, and Age"
    )

    # Show the plot
    return fig

def plot_pie_demographic(expanded_df, col='Gender'):
    # Set Gender column from IsMale
    #expanded_df = expanded_df.rename(columns={'IsMale': 'Gender'})
    #expanded_df['Gender'] = expanded_df['Gender'].map({True: 'Male', False: 'Female'})

    temp = expanded_df.groupby(col, as_index=0).size().sort_values(by='size', ascending=False)
    temp = temp.reset_index(drop=True)
    temp.loc[20:, col] = 'Other'
    temp[col] = temp[col].apply(lambda s: s[:28]+'...' if len(s) > 20 else s)
    fig = px.pie(temp, values='size', names=col, title='Count of ' + col)

    return fig

def plot_bar_demographic(expanded_df, col='Age'):
    # Custom order for age groups
    age_group_order = ["0-2", "3-6", "7-12", "13-18", "19-24", "25-34", 
                       "35-44", "45-54", "55-64", "65-74", "75 or over"]
    time_using_drug_order = ['less than 1 month', '1 to 6 months', '6 months to less than 1 year', 
                             '1 to less than 2 years', '2 to less than 5 years', '5 to less than 10 years', '10 years or more']
    order_dict = {'Age': age_group_order, 'TimeUsingDrug': time_using_drug_order}

    # Reorder the DataFrame based on your custom age group order
    expanded_df = expanded_df.fillna(' ')
    expanded_df = expanded_df[expanded_df[col]!=' ']
    expanded_df[col] = pd.Categorical(expanded_df[col], categories=order_dict[col], ordered=True)
    df_age = expanded_df[col].value_counts().sort_index().reset_index()
    df_age.columns = [col, 'counts']

    # Create a bar plot
    fig = px.bar(x=df_age[col], y=df_age['counts'])

    # Update layout for a better look
    fig.update_layout(
        #height=300, width=600, 
        showlegend=True, title_text="Count of " + col, 
        xaxis_title=col, yaxis_title='Count'
    )

    # Show the plot
    return fig

########################################
# Show the distribution of comment lengths
def comment_length_histgram(expanded_df):
    expanded_df['CommentLength'] = expanded_df['Comment'].apply(lambda x: len(x.split()))

    expanded_df.groupby(['CommentLength'], as_index=0, dropna=0).size().sort_values('CommentLength', ascending=1)[:30]

    fig = px.histogram(expanded_df, x='CommentLength', color='bert_compound_threshold')
    # Update layout for a better look
    fig.update_layout(
        title_text="Review lengths by Sentiment", 
        xaxis_title="Review Length", yaxis_title="Count"
    )

    return fig

########################################
# Show the sentiment for each family
def plot_sentiment_by_family(expanded_df, model):
    grouped1 = expanded_df.groupby(['DrugFamily'], as_index=0, dropna=0).agg({'Overall': 'mean'})
    grouped2 = expanded_df.groupby(['DrugFamily'], as_index=0, dropna=0).agg({model+'_compound': 'mean'})
    grouped = pd.merge(grouped1, grouped2, on=['DrugFamily'], how='left')
    grouped = grouped.sort_values(['Overall'], ascending=False)

    # Initialize a figure
    fig = go.Figure()

    # Add bar plots for each rating type
    for rating in [model+'_compound']:
        fig.add_trace(go.Bar(
            x=grouped['DrugFamily'],
            y=grouped[rating],
            name=rating
        ))

    # Adjusting the layout for better visualization
    fig.update_layout(
        barmode='group',
        title=model+' Compound Sentiment by Drug Family <br><sup>We see a similar trend as the actual Ratings in terms of Drug Families</sup>',
        xaxis_title='Drug Family',
        yaxis_title=model+' Sentiment',
        legend_title='Rating Type'
    )
    # Show the plot
    #fig.update_layout(yaxis_range=[-1,1])
    return fig

########################################
# Show the sentiment vs rating
def compound_sentiment_vs_rating_boxplot(expanded_df, model):
    plot_df = expanded_df.round(2)
    plot_df = plot_df[plot_df[model+'_compound'].notna()]

    # Create a box plot with Plotly Express
    fig = px.box(plot_df, x="Overall", y=model+"_compound", title='Overall Sentiment Score vs Overall Rating')

    # Add vertical lines to your Plotly figure
    fig.add_vline(x=2.5, line_color="red", line_dash="dash")
    fig.add_vline(x=3.833, line_color="red", line_dash="dash")

    # Update layout for a better look
    fig.update_layout(
        #height=300, width=400, 
        showlegend=True, 
        title_text=model+' Sentiment vs Actual Ratings <br><sup>We see sentiment goes up as Review Rating goes up, as expected</sup>',
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text="Actual Rating<br><sup>The red dotted lines categorise the reviews into three categories: (Negative, Neutral, Positive)</sup>"
            )
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text=model+' Sentiment'
            )
        )
    )

    # Show the figure
    return fig

def individual_sentiments_vs_rating_boxplot(expanded_df, model):
    plot_df = expanded_df.round(2)
    plot_df = plot_df[plot_df[model+'_compound'].notna()]

    # Assuming plot_df is your DataFrame and model is the model name as a string
    fig = make_subplots(rows=1, cols=3, subplot_titles=(f'{model}_neg', f'{model}_neu', f'{model}_pos'))

    # Negative sentiment box plot
    fig.add_trace(
        go.Box(y=plot_df[model+'_neg'], x=plot_df['Overall'], name='Negative', marker_color='red'),
        row=1, col=1
    )
    # Neutral sentiment box plot
    fig.add_trace(
        go.Box(y=plot_df[model+'_neu'], x=plot_df['Overall'], name='Neutral', marker_color='blue'),
        row=1, col=2
    )
    # Positive sentiment box plot
    fig.add_trace(
        go.Box(y=plot_df[model+'_pos'], x=plot_df['Overall'], name='Positive', marker_color='green'),
        row=1, col=3
    )
    # Update layout for a better look
    fig.update_layout(
        #height=500, width=1200, 
        title_text=f'''Individual Sentiment Scores vs Overall Rating
                       <br><sup>BERT returns three scores: (negativity, neutrality, positivity) in a sentiment analysis. This plot shows their relation with the actual ratings</sup>''',
    )

    return fig

def sentiment_vs_rating_categories_boxplot(expanded_df, model):
    # Filter out rows with NaN compound scores
    plot_df = expanded_df.round(2)
    plot_df = plot_df[plot_df[model+'_compound'].notna()]

    # Initialize a figure with 1 row and 3 columns
    fig = make_subplots(rows=1, cols=3, subplot_titles=('Satisfaction', 'EaseOfUse', 'Effectiveness'))

    # Add boxplot for Satisfaction vs compound scores
    fig.add_trace(
        go.Box(y=plot_df[model+'_compound'], x=plot_df['Satisfaction'], name='Satisfaction'),
        row=1, col=1
    )
    # Add boxplot for EaseOfUse vs compound scores
    fig.add_trace(
        go.Box(y=plot_df[model+'_compound'], x=plot_df['EaseOfUse'], name='EaseOfUse'),
        row=1, col=2
    )
    # Add boxplot for Effectiveness vs compound scores
    fig.add_trace(
        go.Box(y=plot_df[model+'_compound'], x=plot_df['Effectiveness'], name='Effectiveness'),
        row=1, col=3
    )

    # Update layout for a better look
    fig.update_layout(
        #height=500, width=1200, 
        title_text=model + ' Sentiment vs Rating Categories',
    )

    return fig

########################################
# read in bertopic data based on family and sentiment chosen
def read_bertopic(family, sentiment):
    name = family + '_' + sentiment
    bertopic_path = '../data/bertopic'
    path = bertopic_path+ '/' +name

    with open(bertopic_path+'/'+'json_out.json', 'r') as f:
        loaded_results = json.load(f)

    # Example: Loading a DataFrame
    #topic_df = pd.read_csv(loaded_results[name]['df'])
    topic_df = pd.read_json(loaded_results[name]['df'], orient='records', lines=True)
    # Example: Loading a spaCy model
    topic_model = BERTopic.load(loaded_results[name]['model'])

    return family, sentiment, topic_df, topic_model

########################################
# Plots the topics by demographic chosen (e.g. IsMale)
def group_topics_by_demographic(topic_df, group='IsMale'):
    topic_list_col = 'TopicNames'

    def reorder_all_category_cols(df):
        try:
            # Define the custom order
            order = ['less than 1 month', '1 to 6 months', '6 months to less than 1 year',
                    '1 to less than 2 years', '2 to less than 5 years','5 to less than 10 years',
                    '10 years or more']
            df['TimeUsingDrug'] = pd.Categorical(df['TimeUsingDrug'], categories=order, ordered=True)
            df = df.sort_values('TimeUsingDrug')
        except:
            pass

        return df

    # Explode the 'list_column' so each item in the list becomes a row
    exploded_df = topic_df.explode(topic_list_col)

    # Now group by 'group_column' and get the value counts for 'list_column'
    counts_df = exploded_df.groupby(group)[topic_list_col].value_counts().reset_index(name='counts')

    # Calculate total size for each gender group
    total_size_by_gender = topic_df.groupby(group, as_index=0).size()
    # Normalize size within each gender group
    counts_df = pd.merge(counts_df, total_size_by_gender, on=group)
    counts_df['perc_count'] = counts_df['counts'] / counts_df['size']

    counts_df = counts_df.sort_values(by=[group,topic_list_col], ascending=True)
    counts_df[topic_list_col] = counts_df[topic_list_col].astype(str)

    counts_df = reorder_all_category_cols(counts_df)
    counts_df = counts_df.sort_values([group,topic_list_col])

    # Now you have a DataFrame with the counts of each item in 'list_column' grouped by 'group_column'
    # Create a bar chart using Plotly Express
    fig = px.bar(
        counts_df, 
        color=topic_list_col, y='perc_count', x=group, barmode='group',
        title='Counts of Items Grouped by Another Column')
    fig.update_yaxes(tickangle=45)

    return fig

# Get the correlation between topics and demographics (e.g. Females complain more about weight gain)
'''def calculate_topic_correlations(topic_df, demographics=['IsMale']):

    # Assuming 'df' is your DataFrame with a binary representation of topics and numeric demographics

    def reorder_all_category_cols(df):
        # Define the custom order
        df.index = pd.Categorical(df.index, categories=time_using_drug_order, ordered=True)
        df = df.sort_index()

        return df
    
    # Convert categorical demographics to numeric if necessary
    # For example, using one-hot encoding for gender
    df = pd.get_dummies(topic_df[demographics+['Topic']], columns=demographics+['Topic'], drop_first=0, prefix='',prefix_sep='')
    demo_cols = pd.get_dummies(topic_df[demographics], columns=demographics, drop_first=0, prefix='',prefix_sep='').columns
    topic_cols = pd.get_dummies(topic_df[['Topic']], columns=['Topic'], drop_first=0, prefix='',prefix_sep='').columns

    # Calculate correlations
    correlations = {}
    for topic in topic_cols:  # topic_list is a list of your topic column names
        topic_correlations = {}
        for demo in demo_cols:  # demographic_list is a list of your demographic column names
            if (df[demo].dtype == 'bool') or (len(df[demo].unique()) == 2):
                corr, _ = pointbiserialr(df[topic], df[demo])
            else:
                corr, _ = spearmanr(df[topic], df[demo])
            topic_correlations[demo] = corr
        correlations[topic] = topic_correlations

    # Convert to DataFrame for easier viewing
    correlation_df = pd.DataFrame(correlations)

    # Reorder the TimeUsingDrug column
    if 'TimeUsingDrug' in demographics:
        correlation_df = reorder_all_category_cols(correlation_df)

    # Create a heatmap
    fig = px.imshow(correlation_df,
                    x=correlation_df.columns,
                    y=correlation_df.index,
                    labels=dict(x="Topic", y=demographics[0])
                    )
    
     # Print the correlation matrix
    return(fig)'''

# Get the correlation between topics and demographics (e.g. Females complain more about weight gain)
def calculate_topic_correlations(topic_df, demographics=['IsMale']):

    # Assuming 'df' is your DataFrame with a binary representation of topics and numeric demographics

    def reorder_all_category_cols(df):
        # Define the custom order
        df.index = pd.Categorical(df.index, categories=time_using_drug_order, ordered=True)
        df = df.sort_index()

        return df
    
    # Get the topics and their probabilities
    topic_cols = topic_df['TopicProbability'].apply(pd.Series)
    topic_cols.columns = [f'Topic{i}' for i in range(len(topic_cols.columns))]
    df = pd.get_dummies(topic_df[demographics], columns=demographics, drop_first=0, prefix='',prefix_sep='')
    demo_cols = df.columns
    df = pd.concat([df, topic_cols], axis=1)
    
    # Convert categorical demographics to numeric if necessary
    # For example, using one-hot encoding for gender
    #df = pd.get_dummies(topic_df[demographics+['Topic']], columns=demographics+['Topic'], drop_first=0, prefix='',prefix_sep='')
    #demo_cols = pd.get_dummies(topic_df[demographics], columns=demographics, drop_first=0, prefix='',prefix_sep='').columns
    #topic_cols = pd.get_dummies(topic_df[['Topic']], columns=['Topic'], drop_first=0, prefix='',prefix_sep='').columns

    # Calculate correlations
    correlations = {}
    for topic in topic_cols:  # topic_list is a list of your topic column names
        topic_correlations = {}
        for demo in demo_cols:  # demographic_list is a list of your demographic column names
            if (df[demo].dtype == 'bool') or (len(df[demo].unique()) == 2):
                corr, _ = pointbiserialr(df[topic], df[demo])
            else:
                corr, _ = spearmanr(df[topic], df[demo])
            topic_correlations[demo] = corr
        correlations[topic] = topic_correlations

    # Convert to DataFrame for easier viewing
    correlation_df = pd.DataFrame(correlations)

    # Reorder the TimeUsingDrug column
    if 'TimeUsingDrug' in demographics:
        correlation_df = reorder_all_category_cols(correlation_df)

    # Create a heatmap
    fig = px.imshow(correlation_df,
                    x=correlation_df.columns,
                    y=correlation_df.index,
                    labels=dict(x="Topic", y=demographics[0])
                    )
    
     # Print the correlation matrix
    return(fig)

# plot the individual reviews
def visualise_documents_custom(topic_df):
    # Reduce the topic probabilities to 2D
    df = topic_df[['UMAP_x', 'UMAP_y']].rename(columns={'UMAP_x': 'x', 'UMAP_y': 'y'})

    # Add your topics to the DataFrame
    df['topic'] = topic_df['Topic']  # Assuming 'topics' is a list of the topic for each document

    # Map the topic labels to the topics
    df['topic_name'] = topic_df['TopicName']

    # Add your Comment to the DataFrame
    df['Comment'] = topic_df['Comment']
    df['Comment'] = df['Comment'].apply(lambda x: (x[:90] + '...') if len(x) > 75 else x)

    df['probability'] = topic_df['TopicProbability'].apply(max)

    df['Id'] = topic_df['Id']

    # Map topic names to integers
    unique_topic_names = list(set(df['topic']))  # `colors` is the list of topic names you have
    topic_name_to_int = {name: i for i, name in enumerate(unique_topic_names)}
    # Add a column for color which maps the topic names to integers
    df['color'] = df['topic'].map(topic_name_to_int)

    # Concatenate the strings from multiple columns to form the hover text
    df['hover_text'] = 'Review: ' + df['Comment'].astype(str) + '<br>' + \
                    'Topic: ' + df['topic_name'].astype(str) + '<br>' + \
                    'Probability: ' + df['probability'].astype(str)

    # Create a scatter plot
    fig = go.Figure(data=go.Scatter(
        x=df['x'],
        y=df['y'],
        mode='markers',
        marker=dict(
            size=5,
            color=df['color'],  # Use the integers from the 'color' column for the colors
            colorscale='Viridis',  # This is one of Plotly's built-in color scales
            showscale=True  # This shows the color scale alongside the scatter plot
        ),
        text=df['hover_text'],  # This will show the topic name when you hover over a point
        hoverinfo='text',  # Only show the text on hover
        hovertemplate='<b>%{text}</b><extra></extra>',  # Custom hover template with bold text
        meta=df['Id'],
    ))

    # Update layout for a better look
    fig.update_layout(
        title='2D UMAP Projection of Document-Topic Probabilities',
        xaxis_title='UMAP Dimension 1',
        yaxis_title='UMAP Dimension 2'
    )

    # Show the plot
    return fig

def create_topic_text(topic_model):
    topics_html = list()
    for topic_html in [
        html.Span([str(r['Topic']) + ": " + ' '.join(r['Representation'])], style={"color": col_swatch[i%10]})
        for i, r in topic_model.get_topic_info().iterrows()
    ]:
        topics_html.append(topic_html)
        topics_html.append(html.Br())

    return topics_html

########################################
def create_placeholder_figure(message="Visualization not available"):
    """
    Create a placeholder figure with a given message.
    """
    fig = go.Figure()
    # Add dummy trace to create an empty plot
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                             marker=dict(color='rgba(0,0,0,0)')))
    # Update layout with the message
    fig.update_layout(
        title=message, xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig
########################################

def plot_ratings_vs_predicted_rating(review):
    # Initialize a figure
    fig = go.Figure()

    # Define x positions for the actual ratings and the predicted rating
    x_actual_ratings = [0, 0.8, 1.6]  # Positions for MeanEaseOfUse, MeanEffectiveness, MeanSatisfaction
    x_predicted_rating = 4  # Position for PredictedOverall, leaving a gap after the actual ratings
    
    bar_width = 0.8  # Width of each actual rating bar
    predicted_bar_width = bar_width * 3  # Making the predicted rating bar width equal to the total width of the three bars

    # Add bar plots for each rating type
    ratings = ['Effectiveness', 'EaseOfUse', 'Satisfaction']
    for i, rating in enumerate(ratings):
        fig.add_trace(go.Bar(
            x=[x_actual_ratings[i]],
            y=[int(review[rating])],
            name=rating,
            width=bar_width
        ))

    # Add predicted rating
    fig.add_trace(go.Bar(
        x=[x_predicted_rating],
        y=[float(review['PredictedOverall'])],
        name='Predicted Rating',
        width=predicted_bar_width
    ))

    # Adjusting the layout for better visualization
    fig.update_layout(
        height=400, width=600,
        barmode='group',
        title='Ratings vs Predicted Rating',
        xaxis_title='Type',
        yaxis_title='Ratings',
        legend_title='Rating Type',
        xaxis=dict(tickvals=[1, 4], ticktext=['Actual Ratings', 'Predicted Rating']),  # Custom tick labels
        yaxis_range=[0, 5]
    )

    # Add horizontal line for Overall values
    fig.add_shape(type='line',
                  x0=-0.4,  # Start slightly before the first bar
                  y0=float(review['Overall']),
                  x1=2,  # End at the middle of the predicted rating bar
                  y1=float(review['Overall']),
                  line=dict(color='black', dash='dash'),
                  xref='x',
                  yref='y')

    # Add Overall value as annotation
    fig.add_annotation(
        x=1, y=float(review['Overall']),
        text="Actual Overall: " + str(round(float(review['Overall']), 2)),
        showarrow=False,
        yshift=10)

    # Show the plot
    return fig

########################################

def epoch_to_datetime(epoch):
    return datetime.datetime.fromtimestamp(epoch/1000)

def add_date_cols(df):
    df['YearPosted'] = df['DatePosted'].apply(lambda d: epoch_to_datetime(d).year)
    df['MonthPosted'] = df['DatePosted'].apply(lambda d: epoch_to_datetime(d).month)
    df['YearMonthPosted'] = df['YearPosted'].astype(str) + '-' + df['MonthPosted'].astype(str)
    df['DatePosted'] = df['DatePosted'].apply(epoch_to_datetime)

    return df

########################################
# Show the rating stats of the reviews
def calculate_stats(expanded_df):
    stats_df = expanded_df[ratings].describe().reset_index()
    stats_df.columns = ['Statistic'] + ratings
    stats_df = round(stats_df, 2)
    stats_df = stats_df[stats_df['Statistic'].isin(['mean','25%', '50%', '75%'])]

    return stats_df

########################################
    
# read in the data
#df = pd.read_json('../data/2_webmd_preprocessed_sentiment.csv', orient='records', lines=True)
print('reading from s3')
df = load_from_s3('maxim-thesis', 'data/2_webmd_preprocessed_sentiment.csv')
# Expand the df into single reviews
expanded_df = expand_reviews(df)

# Set Gender column from IsMale
expanded_df = expanded_df.rename(columns={'IsMale': 'Gender'})
expanded_df['Gender'] = expanded_df['Gender'].map({True: 'Male', False: 'Female'})

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

body_layout = \
dbc.Container([
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Exploratory Data Analysis', value='tab-eda', children=[
            dbc.Row([
                dbc.Col([
                    dcc.Markdown(
                        f"""
                        -----
                        ## Exploratory Drug data visualisation from WebMD
                        -----
                        A general overview of the ratings by drug family, demographics, and comment length.
                        """
                    )
                ],
                    sm=12,
                    md=8,
                ),
                dbc.Col([
                    dcc.Markdown(
                        f"""
                        -----
                        ##### Data:
                        -----
                        For this research, psychiatric drug reviews from
                        [WebMD](https://www.webmd.com/drugs/2/index) were collected.

                        These psychiatric drugs are categorised based on their drug family.
                        
                        """
                    )
                ],
                    sm=12,
                    md=4,
                ),
            ]),

            dbc.Row([
                dcc.Markdown(
                        f"""
                        -----
                        """
                )
            ]),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        id='ratings_by_family',
                        figure=plot_ratings_by_family(expanded_df)
                    ),
                ]),
            ]),

            dbc.Row([
                dcc.Markdown(
                        f"""
                        -----
                        """
                )
            ]),

            html.Div([
                dbc.Row([
                    dbc.Col([
                        dbc.Badge(
                            "Drug Family:", color="info", className="mr-1"
                        ),
                        dcc.Dropdown(
                            id='drug-family-dropdown',
                            options=[{'label': family, 'value': family} for family in drug_families],
                            value='All',  # Default value
                            #placeholder="Drug Famil(ies) ...",
                            multi=True,
                            clearable=False,
                        ),
                    ],
                        sm=12,
                        md=2,
                    ),

                    dbc.Col([
                        dbc.Badge(
                            "Gender:", color="info", className="mr-1"
                        ),
                        dcc.Dropdown(
                            id='gender-dropdown',
                            options=[{'label': family, 'value': family} for family in list(set(expanded_df['Gender'].dropna()))],
                            value= list(set(expanded_df['Gender'].dropna())), #list(set(expanded_df['IsMale'])),  # Default value
                            #placeholder="Drug Famil(ies) ...",
                            multi=True,
                            clearable=False,
                        ),
                    ],
                        sm=12,
                        md=2,
                    ),

                    dbc.Col([
                        dbc.Badge(
                            "Age Group:", color="info", className="mr-1"
                        ),
                        dcc.Dropdown(
                            id='age-dropdown',
                            options=[{'label': family, 'value': family} for family in ['All'] + sorted(list(set(expanded_df['Age'].dropna())))],
                            value= 'All', #list(set(expanded_df['IsMale'])),  # Default value
                            #placeholder="Drug Famil(ies) ...",
                            multi=True,
                            clearable=False,
                        ),
                    ],
                        sm=12,
                        md=2,
                    ),

                    dbc.Col([
                        dbc.Badge(
                            "Time Using Drug:", color="info", className="mr-1"
                        ),
                        dcc.Dropdown(
                            id='time-dropdown',
                            options=[{'label': family, 'value': family} for family in ['All'] + time_using_drug_order],
                            value= 'All', #list(set(expanded_df['IsMale'])),  # Default value
                            #placeholder="Drug Famil(ies) ...",
                            multi=True,
                            clearable=False,
                        ),
                    ],
                        sm=12,
                        md=2,
                    ),
                ],
                    justify="center"
                ),
            ],
                style={'position': 'sticky', 'top': 0, 'zIndex': 1000, 'background': '#fff', 'padding': '10px', 'borderBottom': '1px solid #ddd'}
            ),

            dbc.Row([
                dbc.Col([
                    dash_table.DataTable(
                        id='statistics-table',
                        columns=[{"name": i, "id": i} for i in ['Statistic'] + ratings],
                        style_cell={'textAlign': 'left'},
                        style_header={
                            'backgroundColor': 'rgb(230, 230, 230)',
                            'fontWeight': 'bold'
                        },
                        style_data_conditional=[
                            {
                                'if': {'column_id': 'Statistic'},
                                'backgroundColor': 'rgb(230, 230, 230)',  # Light blue background
                                'fontWeight': 'bold'
                            },
                            {
                                'if': {'column_id': 'Overall'},
                                'fontWeight': 'bold'
                            }
                        ]
                    )
                ],
                    sm=12,
                    md=8,
                ),
            ],
                justify="center"
            ),

            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='rating-distribution'),
                ]),
                dbc.Col([
                    dcc.Graph(id='comment-length'),
                ]),
            ]),

            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='demographic-fig1'),
                ]),
                dbc.Col([
                    dcc.Graph(id='demographic-fig2'),
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='demographic-fig3'),
                ]),
                dbc.Col([
                    dcc.Graph(id='demographic-fig4'),
                ]),
            ]),
        
        ]),

        dcc.Tab(label='Sentiment Analysis', value='tab-sentiment', children=[
            dbc.Row([
                dbc.Col([
                    dcc.Markdown(
                        f"""
                        -----
                        ## Drug Sentiment Analysis
                        -----
                        The sentiment analysis results and how they relate with the customer ratings
                        """
                    )
                ],
                    sm=12,
                    md=8,
                ),
                dbc.Col([
                    dcc.Markdown(
                        f"""
                        -----
                        ##### Data:
                        -----
                        For this research, psychiatric drug reviews from
                        [WebMD](https://www.webmd.com/drugs/2/index) were collected.

                        These psychiatric drugs are categorised based on their drug family.
                        
                        """
                    )
                ],
                    sm=12,
                    md=4,
                ),
            ]),

            dbc.Row([
                dcc.Markdown(
                        f"""
                        -----
                        """
                )
            ]),

            # Sentiment by family
            # Compound sentiment vs rating
            # Individual Sentiments vs rating
            # Sentiment vs rating categories
            # 

            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        id='sentiment-by-family',
                        figure=plot_sentiment_by_family(expanded_df, 'bert')
                    ),
                ],
                    sm=12,
                    md=5,
                ),
                dbc.Col([
                    dcc.Graph(
                        id='sentiment-vs-rating-categories',
                        figure=sentiment_vs_rating_categories_boxplot(expanded_df, 'bert')
                    ),
                ],
                    sm=12,
                    md=7,
                ),
            ]),

            dbc.Row([
                dcc.Markdown(
                        f"""
                        -----
                        """
                )
            ]),

            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        id='compound-sentiment-vs-rating',
                        figure=compound_sentiment_vs_rating_boxplot(expanded_df, 'bert')
                    ),
                ]),
            ]),

            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        id='individual-sentiments-vs-rating',
                        figure=individual_sentiments_vs_rating_boxplot(expanded_df, 'bert')
                    ),
                ]),
            ]),
        ]),

        dcc.Tab(label='Topic Modelling', value='tab-topic', children=[
            dbc.Row([
                dbc.Col([
                    dcc.Markdown(
                        f"""
                        -----
                        ## Topic Modelling Results
                        -----
                        This section shows the topics generated by the BERTopic model for each Drug Family and Sentiment. \\
                        Use the dropdowns to select the Drug Family and Sentiment to view the topics.
                        """
                    )
                ],
                    sm=12,
                    md=8,
                ),
                dbc.Col([
                    dcc.Markdown(
                        f"""
                        -----
                        ##### Data:
                        -----
                        For this research, psychiatric drug reviews from
                        [WebMD](https://www.webmd.com/drugs/2/index) were collected.

                        These psychiatric drugs are categorised based on their drug family.
                        
                        """
                    )
                ],
                    sm=12,
                    md=4,
                ),
            ]),

            dbc.Row([
                dcc.Markdown(
                        f"""
                        -----
                        """
                )
            ]),

            html.Div([
                dbc.Row([
                    dbc.Col([
                        dbc.Badge(
                            "Drug Family:", color="info", className="mr-1"
                        ),
                        dcc.Dropdown(
                            id='drug-family-dropdown-topic',
                            options=[{'label': i, 'value': i} for i in DrugFamilies],
                            value=DrugFamilies[0]  # default value
                        ),
                    ],
                        sm=12,
                        md=2,
                    ),

                    dbc.Col([
                        dbc.Badge(
                            "Sentiment:", color="info", className="mr-1"
                        ),
                        dcc.Dropdown(
                            id='sentiment-dropdown-topic',
                            options=[{'label': i, 'value': i} for i in sentiments],
                            value=sentiments[0]  # default value
                        ),
                    ],
                        sm=12,
                        md=2,
                    ),
                ],
                    justify="center"
                ),
            ],
                style={'position': 'sticky', 'top': 0, 'zIndex': 1000, 'background': '#fff', 'padding': '10px', 'borderBottom': '1px solid #ddd'}
            ),
            # Div to hold the plot or table that displays the data
            html.Div(id='data-display'),
            dcc.Store(id='topic-data'),  # This component will store the selected dataset

            dbc.Row([
                dcc.Markdown(
                    """
                    All the words in the topics and their importance are shown in the plot below.
                    """,
                    style={'marginTop': '20px'}  # Adjust '20px' as needed
                )
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='visualize_barchart'),
                ],
                    sm=12,
                    md=8
                ),
                dbc.Col([
                    dbc.Row([
                        dcc.Markdown(
                            """
                                ##### \# Reviews:
                            """
                        ),
                        html.Div(
                            id='num_docs',
                            style={
                                "fontSize": 18,
                                "height": "50px",
                                "overflow": "auto",
                            },
                        ),
                    ]), # Adjust '20px' as needed),
                    dbc.Row([
                        dcc.Markdown(
                            """
                                ##### Topics:
                            """
                        ),
                        html.Div(
                            id='topic_text',
                            style={
                                "fontSize": 11,
                                "height": "200px",
                                "overflow": "auto",
                            },
                        ),
                    ]),
                ],
                    sm=12,
                    md=4
                ),
            ]),
            dbc.Row([
                dcc.Markdown(
                    """
                    -----
                    """,
                )
            ]),
            dbc.Row([
                dcc.Markdown(
                    """
                    A 2D UMAP projection of the document-topic probabilities is shown below.
                    """,
                    style={'marginTop': '20px'}  # Adjust '20px' as needed
                )
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='visualize_topics'),
                ]),
            ]),
            dbc.Row([
                dcc.Markdown(
                        f"""
                        -----
                        """
                )
            ]),
            dbc.Row([
                dcc.Markdown(
                    """
                    A heatmap showing the correlation between topics and demographics is shown below.
                    """,
                    style={'marginTop': '20px'}  # Adjust '20px' as needed
                )
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='visualize_heatmap'),
                ]),
            ]),
            dbc.Row([
                dcc.Markdown(
                        f"""
                        -----
                        """
                )
            ]),
            dbc.Row([
                dcc.Markdown(
                    """
                    The below plots show the distribution of topics by demographics.
                    """,
                    style={'marginTop': '20px'}  # Adjust '20px' as needed
                )
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Row([
                        dcc.Graph(id='topics_by_demographic'),
                    ]),
                    dbc.Row([
                        dcc.Graph(id='topic_correlations'),
                    ]),
                ], 
                    sm=12,
                    md=8
                ),
                dbc.Col([
                    dbc.Row([
                        dbc.Badge(
                            "Group by:", color="info", className="mr-1"
                        ),
                        dcc.Dropdown(
                            id='group-by-dropdown',
                            options=[{'label': family, 'value': family} for family in ['Gender','Age','TimeUsingDrug','Condition']],
                            value='Gender',  # Default value
                            #placeholder="Drug Famil(ies) ...",
                            clearable=False,
                        ),
                    ]),
                ], 
                    sm=12,
                    md=4
                )
            ])
        ]),

        dcc.Tab(label='Individual Review Analysis', value='tab-individual', children=[
            dbc.Row([
                dbc.Col([
                    dcc.Markdown(
                        f"""
                        -----
                        ## Individual Review Analysis
                        -----
                        This section allows us to view individual reviews and their sentiment analysis.
                        """
                    )
                ],
                    sm=12,
                    md=8,
                ),
                dbc.Col([
                    dcc.Markdown(
                        f"""
                        -----
                        ##### Data:
                        -----
                        For this research, psychiatric drug reviews from
                        [WebMD](https://www.webmd.com/drugs/2/index) were collected.

                        These psychiatric drugs are categorised based on their drug family.
                        
                        """
                    )
                ],
                    sm=12,
                    md=4,
                ),
            ]),

            dbc.Row([
                dcc.Markdown(
                    """
                    -----
                    """,
                )
            ]),

            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='visualize_documents'),
                ],
                    sm=12,
                    md=10
                ),
                dbc.Col([

                    dbc.Row([
                        dbc.Badge(
                            "Gender:", color="info", className="mr-1"
                        ),
                        dcc.Dropdown(
                            id='gender-dropdown-individual',
                            options=[{'label': family, 'value': family} for family in list(set(expanded_df['Gender'].dropna()))],
                            value= list(set(expanded_df['Gender'].dropna())), #list(set(expanded_df['IsMale'])),  # Default value
                            #placeholder="Drug Famil(ies) ...",
                            multi=True,
                            clearable=False,
                        ),
                    ]),

                    dbc.Row([
                        dbc.Badge(
                            "Age Group:", color="info", className="mr-1"
                        ),
                        dcc.Dropdown(
                            id='age-dropdown-individual',
                            options=[{'label': family, 'value': family} for family in ['All'] + sorted(list(set(expanded_df['Age'].dropna())))],
                            value= 'All', #list(set(expanded_df['IsMale'])),  # Default value
                            #placeholder="Drug Famil(ies) ...",
                            multi=True,
                            clearable=False,
                        ),
                    ]),

                    dbc.Row([
                        dbc.Badge(
                            "Time Using Drug:", color="info", className="mr-1"
                        ),
                        dcc.Dropdown(
                            id='time-dropdown-individual',
                            options=[{'label': family, 'value': family} for family in ['All'] + time_using_drug_order],
                            value= 'All', #list(set(expanded_df['IsMale'])),  # Default value
                            #placeholder="Drug Famil(ies) ...",
                            multi=True,
                            clearable=False,
                        ),
                    ]),

                    dbc.Row([
                        dbc.Badge(
                            "Condition Using Drug:", color="info", className="mr-1"
                        ),
                        dcc.Dropdown(
                            id='condition-dropdown-individual',
                            options=[{'label': family, 'value': family} for family in ['All'] + sorted(list(set(expanded_df['Condition'].dropna())))],
                            value= 'All', #list(set(expanded_df['IsMale'])),  # Default value
                            #placeholder="Drug Famil(ies) ...",
                            multi=True,
                            clearable=False,
                        ),
                    ]),
                ],
                    sm=12,
                    md=2
                ),
            ]),
            
            dbc.Row([
                dbc.Alert(
                    id="review_summary",
                    children="Click on a Review to see its details here",
                    color="secondary",
                ),
            ]),

            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='ratings_vs_predicted_rating_fig'),
                ]),
                dbc.Col([
                    dcc.Graph(id='topic_distribution'),
                ])
            ]),
        ])
    ]),
    ##############################################
    
    ##############################################

    ##############################################
 
    ##############################################

])

# Define the layout of the app
app.layout = html.Div(children=[body_layout])

# Callback to update the graph based on the dropdown selection
@app.callback(
    Output('rating-distribution', 'figure'),
    Output('statistics-table', 'data'),
    Output('comment-length', 'figure'),
    Output('demographic-fig1', 'figure'),
    Output('demographic-fig2', 'figure'),
    Output('demographic-fig3', 'figure'),
    Output('demographic-fig4', 'figure'),
    [Input('drug-family-dropdown', 'value'),
     Input('gender-dropdown', 'value'),
     Input('age-dropdown', 'value'),
     Input('time-dropdown', 'value')]
)
# Function to update plot
def update_eda_output(drug_family, gender, age, time_using_drug):
    # Filter based on the dropdown selection
    if 'All' not in drug_family:
        df = expanded_df[expanded_df['DrugFamily'].isin(drug_family)]
    else:
        df = expanded_df

    df = df[df['Gender'].isin(gender)]

    if 'All' not in age:
        df = df[df['Age'].isin(age)]
    else:
        df = df

    if 'All' not in time_using_drug:
        df = df[df['TimeUsingDrug'].isin(time_using_drug)]
    else:
        df = df
    
    # Create the plots
    rating_distribution_fig = rating_distribution(df)
    stats_df = calculate_stats(df)
    comment_length_fig = comment_length_histgram(df)

    # These 4 will be subplots together
    demographic_fig1 = plot_pie_demographic(df, col='Gender')
    demographic_fig2 = plot_pie_demographic(df, col='Condition')
    demographic_fig3 = plot_bar_demographic(df, col='Age')
    demographic_fig4 = plot_bar_demographic(df, col='TimeUsingDrug')

    # Show the plot
    return rating_distribution_fig, stats_df.to_dict('records'), comment_length_fig, \
    demographic_fig1, demographic_fig2, demographic_fig3, demographic_fig4

# Callback to load data based on dropdown selection
@app.callback(
    Output('visualize_topics', 'figure'),
    Output('visualize_barchart', 'figure'),
    Output('num_docs', 'children'),
    Output('topic_text', 'children'),
    Output('visualize_heatmap', 'figure'),
    Output('visualize_documents', 'figure'),
    Output('topics_by_demographic', 'figure'),
    Output('topic_correlations', 'figure'),
    [Input('drug-family-dropdown-topic', 'value'),
     Input('sentiment-dropdown-topic', 'value'),
     Input('group-by-dropdown', 'value'),
     Input('gender-dropdown-individual', 'value'),
     Input('age-dropdown-individual', 'value'),
     Input('time-dropdown-individual', 'value'),
     Input('condition-dropdown-individual', 'value')]
)
def update_topic_output(drug_family, sentiment, group_by,
                        gender, age, time_using_drug, condition):
    
    # Load the dataset
    family, sentiment, topic_df, topic_model = read_bertopic_from_s3(drug_family, sentiment)

    # Set Gender column from IsMale
    topic_df = topic_df.rename(columns={'IsMale': 'Gender'})
    topic_df['Gender'] = topic_df['Gender'].map({1: 'Male', 0: 'Female'})
    print(topic_df['Gender'].value_counts())
    
    # Create a figure to display the data (this is just an example using a table)
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(topic_df.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[topic_df[col] for col in topic_df.columns],
                   fill_color='lavender',
                   align='left'))
    ])
    try:
        visualize_topics = topic_model.visualize_topics()
    except:
        visualize_topics = create_placeholder_figure(message="Visualization not available")
    visualize_barchart = topic_model.visualize_barchart()
    visualize_barchart.update_layout(
        height=400, width=800, 
        #title_text="gghghgh", showlegend=False
    )
    num_docs = topic_df.shape[0]
    topic_text = create_topic_text(topic_model)
    visualize_heatmap = topic_model.visualize_heatmap()
    #visualize_documents = topic_model.visualize_documents(topic_df['Comment'].tolist(), reduced_embeddings=topic_model.umap_model.embedding_[:, :2],
    #                            hide_document_hover=0, hide_annotations=True)

    # Filter based on the dropdown selection
    df = topic_df[topic_df['Gender'].isin(gender)]
    if 'All' not in age:
        df = df[df['Age'].isin(age)]

    if 'All' not in time_using_drug:
        df = df[df['TimeUsingDrug'].isin(time_using_drug)]

    if 'All' not in condition:
        df = df[df['Condition'].isin(condition)]

    visualize_documents = visualise_documents_custom(df)
    topics_by_demographic = group_topics_by_demographic(topic_df, group=group_by)
    topic_correlations = calculate_topic_correlations(topic_df, demographics=[group_by])

    # Return the figure to be displayed in the 'data-display' Div
    return visualize_topics, visualize_barchart, num_docs, topic_text, visualize_heatmap, visualize_documents, \
        topics_by_demographic, topic_correlations

# Callback to load data based on dropdown selection
@app.callback(
    Output('ratings_vs_predicted_rating_fig', 'figure'),
    Output('topic_distribution', 'figure'),
    Output('review_summary', 'children'),
    [Input('visualize_documents', 'clickData'),
     Input('drug-family-dropdown-topic', 'value'),
     Input('sentiment-dropdown-topic', 'value')]
)
def update_individual_review_output(clickData, choice1_value, choice2_value):
    
    # Load the dataset
    family, sentiment, topic_df, topic_model = read_bertopic_from_s3(choice1_value, choice2_value)
    topic_df = add_date_cols(topic_df)
    contents ="Click on a node to see its details here"

    if clickData is not None:
        review_id = clickData['points'][0]['meta']

        review = topic_df[topic_df['Id']==review_id]
        ratings_vs_predicted_rating_fig = plot_ratings_vs_predicted_rating(review)
        topic_distribution = topic_model.visualize_distribution(pd.Series(review['TopicProbability'].iloc[0]))
        topic_distribution.update_layout(
            height=400, width=600,
        )

        # Set customer variables
        gender = 'Male' if review["IsMale"].iloc[0] == 1 else 'Female'

        contents = []

        contents.append(
            html.Div([
                html.Img(
                    src='/assets/user.png', className='rounded-circle', 
                    style={'height': '2%', 'width': '2%', 'top':'10px'}
                ),
                html.Div(
                    [
                        html.P(
                            gender + " | "
                            + str(review["Age"].iloc[0]) + " | "
                            + str(review["Condition"].iloc[0]) + " | "
                            + str(review["TimeUsingDrug"].iloc[0]),
                            className='align-items-left'
                        )
                    ],
                    style={'flex': '1'}
                ),
                html.Div(
                    [
                        html.P(
                            "Date Posted: " + str(review["DatePosted"].iloc[0]), className='text-muted align-items-right'
                        )
                    ],
                    style={'textAlign': 'right'}
                )
            ], className='d-flex justify-content-between')
        )

        contents.append(
            html.H4("Drug: " + review["Drug"].iloc[0],
                    style={'margin': '10px 0 10px 0'})
        )
        contents.append(
            html.H5("Overall Rating: " + str(round(review["Overall"].iloc[0], 2)),
            style={'margin': '5px 0 5px 0'})
        )
        contents.append(
            html.H6(
                "Effectiveness: " + str(review["Effectiveness"].iloc[0]) +
                " | Ease of Use: " + str(review["EaseOfUse"].iloc[0]) +
                " | Satisfaction: " + str(review["Satisfaction"].iloc[0]) 
            )
        )

        contents.append(
            html.P(
                str(review["Comment"].iloc[0]),
                className='text-muted'
            )
        )

        contents.append(
            html.Div([
                # Thumbs up image
                html.Img(
                    src='/assets/thumbs_up.png',
                    className='rounded-circle',
                    style={'height': '20px', 'width': '20px'}  # Adjust height and width as needed
                ),
                # Number found helpful
                html.Div(
                    [
                        html.P(
                            str(review["NumFoundHelpful"].iloc[0]),
                            style={'margin': '0 10px 0 5px', 'color': 'grey'}  # Adjust margins to control spacing
                        )
                    ],
                    className='d-flex align-items-center'
                ),
                # Thumbs down image
                html.Img(
                    src='/assets/thumbs_down.png',
                    className='rounded-circle',
                    style={'height': '20px', 'width': '20px'}  # Adjust height and width as needed
                ),
                # Number not found helpful
                html.Div(
                    [
                        html.P(
                            str(review["NumVoted"].iloc[0]-review["NumFoundHelpful"].iloc[0]),
                            style={'margin': '0 10px 0 5px', 'color': 'grey'}  # Adjust margins to control spacing
                        )
                    ],
                    className='d-flex align-items-center'
                )
            ], className='d-flex justify-content-start')
        )

            
        return ratings_vs_predicted_rating_fig, \
            topic_distribution, \
            contents
    
    return \
        create_placeholder_figure(), \
        create_placeholder_figure(), \
        contents


# Run the server
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
