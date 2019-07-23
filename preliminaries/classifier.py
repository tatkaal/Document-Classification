import spacy
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import os

from scrapper import build_dataset
from normalizer import normalize_corpus

seed_urls = ['https://inshorts.com/en/read/technology',
             'https://inshorts.com/en/read/sports',
             'https://inshorts.com/en/read/world']

news_df = build_dataset(seed_urls)
# print(news_df.news_category.value_counts())

# combining headline and article text
news_df['full_text'] = news_df["news_headline"].map(str)+ '. ' + news_df["news_article"]

# pre-process text and store the same
news_df['clean_text'] = normalize_corpus(news_df['full_text'])
norm_corpus = list(news_df['clean_text'])

print(news_df[['full_text', 'clean_text']].to_dict()) # show a sample news article
