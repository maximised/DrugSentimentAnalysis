# Cloud-Powered Analytics: Evaluating Psychiatric Drug Efficacy through Customer Sentiment and Clinical Data
This is the code repository for our Cloud Dashboard for the Master's thesis of MTU Cloud Computing. 

## Description

Psychiatric drugs are inherently complex and can affect individuals in a variety of ways.
Analyzing these effects after a drug has been developed can be extremely beneficial.
Online forums provide a venue for patients to discuss their personal experiences with
particular drugs, which can improve patient care and assist researchers in their ongoing
investigations. There are two goals of this study. The first goal is to gather and analyze
patient feedback from WebMD on certain drugs. The aim is to determine if there is
a significant correlation between the sentiment derived from WebMD feedback and the
overall rating the customers give the drugs. The second goal is to determine the key
points of each drug based on the topics identified in customer reviews. The reason
that a customer rated a drug a certain way will be identified. The study draws upon
data from WebMD, where patient feedback is subjected to sentiment analysis and topic
modelling. To facilitate this, a robust and scalable system will be developed using
Docker for containerization and deployed on Amazon Web Services (AWS) to ensure
high availability and efficient resource management.

## Components
- WebMD Scraper: A web scraper that collects reviews from WebMD.
- Preprocess: Scripts for
  1. cleaning and preprocessing the scraped data
  2. Apply Sentiment models onto data using RoBERTa and VADER models.
  3. Apply Topic Models on data using BERTopic
- Dash: A Flask-based web dashboard for visualizing sentiment and topic analysis results.
- Data: A directory that stores the raw and processed datasets along with the models.

## Setup and Installation

### Prerequisites
- Docker
- Python 3.8+
- AWS CLI (for deployment)

### AWS Setup
1. Configure AWS CLI with your credentials:
```bash
aws configure
```

2. Create a S3 bucket in your AWS account called 'maxim-thesis'. This is where the data will be stored.
```bash
aws s3 mb s3://maxim-thesis
```

### Local Setup
1. Clone the repository:
```bash
git clone https://github.com/maximised/DrugSentimentAnalysis.git
```

2. Build the docker images for each component:
i. 

```bash
git clone https://github.com/yourusername/yourprojectname.git
cd yourprojectname
pip install -r requirements.txt
