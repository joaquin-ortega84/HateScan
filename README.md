# Welcome to HateScan 📢

<img width="500" alt="Screenshot 2024-02-29 at 15 07 48" src="https://github.com/joaquin-ortega84/HateScan/assets/104449577/3ff8f7d9-1e05-4cc4-a4ca-2dcedfae33bd">

#

## What is HateScan?

### HateScan offers you the ability to analyze hate speech from real tweets and Twitter accounts, including those of celebrities and public figures. 

HateScan is a web app powered by AI to make classifications of speech in Twitter. HateScan integrates two Reccurent Neural Network models trained on ten's of thousands of records of data to do two primary tasks:

1. Classify a tweet & accounts level of hate:
   - Normal
   - Offensive
   - Hateful
     
3. Classify what the tweet & account topic of the narrative is about:
    - Gender
    - Religion
    - Race
    - Politics
    - Sport
  
<img width="600" alt="Screenshot 2024-02-29 at 16 03 01" src="https://github.com/joaquin-ortega84/Portfolio/assets/104449577/f6e62fe2-cd5e-4940-9ce0-a3d7ec03d86e">

# 

## HateScan Features

### 1. Tweet Scan
Analyze any single tweet and our models will classify its hate label and hate topic (feature now deprecated).

### 2. Account Scan
Analyze any account by inputing Twitter handle and number of tweets wished to analyze. Models return hate label of account and hate topic distribution of the account (feature now only works for accounts in BigQuery DB. Use handles found in the Global Scan chart by hovering over account / data points).

<img width="400" alt="Screenshot 2024-02-29 at 15 37 55" src="https://github.com/joaquin-ortega84/HateScan/assets/104449577/e76c5a44-75ba-4ccf-a190-87a7ecb8ccc8">

### 3. Global Scan
All the accounts analyzed are stored in a Google Cloud's database: BigQuery. We are able to compare all accounts of public figures, politians, artists, world leading figures, athletes. And visualize patterns of speech, through Principal Component Analysis (PCA) and plotting. 

<img width="400" alt="Screenshot 2024-02-29 at 15 37 32" src="https://github.com/joaquin-ortega84/HateScan/assets/104449577/6acb4c29-f341-4fd2-a182-b4be04376877">

Each account is a data point in the graph. The size of the circles represents the number of followers of each profile. The color of the circles represents the hate label assigned by HateScan. Therefore we can see which accounts are most influential based on their following base.

#

### Methodology & Under The Hood

1. Analyzed our initial dataset https://github.com/avaapm/hatespeech
2. Downloaded Tweets text from Tweet IDs using Twitter API in Rapid API
3. Found imbalanced classes so we enriched the classes with additional datasets
4. Trained two Recurrent Neural Network Models
5. Set up our cloud database with BigQuery to store model classification results for first time account scans
6. Built our front-end and app with Streamlit & Fast API
7. Pitched our final project in front of an audience of 100 people given a demo of our solution

#

HateScan web: https://hatescan.streamlit.app/ <br>
Built by Joaquin Ortega, Elina Emsems & Santiago Rodriguez for Le Wagon Data Science Bootcamp Demoy Day (Batch #1237).

