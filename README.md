# Sentiment Analyzer

## Overview
Sentiment Analyzer is a tool designed to analyze and classify the sentiment of text data. It uses natural language processing (NLP) techniques to determine whether the sentiment expressed in a piece of text is positive, negative, or neutral.

## Features
- **Text Preprocessing**: Cleans and prepares text data for analysis.
- **Sentiment Classification**: Classifies text into positive, negative, or neutral sentiment.
- **Visualization**: Provides visual representations of sentiment analysis results.

## Installation
To install the Sentiment Analyzer, clone the repository and install the required dependencies:
```bash
$ git clone https://github.com/randallwstanford/sentiment_analyzer.git

$ cd sentiment_analyzer

$ pip3 install -r requirements.txt
```

## Usage
To use the Sentiment Analyzer, run the following command:
```bash
$ python3 llm.py "this movie sucks"
```

## Training
To train the Sentiment Analyzer on a custom dataset, run the following command:
```bash
$ python3 train.py "data/Train.csv"
```
This will create a trained model named "llm.pt".

## Dataset

The Sentiment Analyzer uses the [IMDb movie reviews dataset](https://www.kaggle.com/datasets/columbine/imdb-dataset-sentiment-analysis-in-csv-format) for training and testing. The dataset contains 50,000 movie reviews labeled as positive or negative sentiment.
