# dataset.py
# Datasets are randomly sampled from public datasets, then serialized to disk to avoid having to re-parse everything.
# Requiers unidecode: %pip install unidecode

import csv
import gzip
import json
import os
import random
import sqlite3
import zipfile
from io import StringIO

from unidecode import unidecode

from task import Task


MAX_SAMPLES_PER_SOURCE = 10

PROCESSED_DATASET_NAME = "saved_test_cases.json.gz"
GROUND_TRUTH_DATASET_NAME = "ground_truth.json"


def get_data():
    # Loads or generates the data.
    data = load_pre_generated_data()
    if data is not None:
        return data
    else:
        return generate_data()


def load_pre_generated_data():
    # If we already did the data processing, load it.
    data = None
    if os.path.isfile(PROCESSED_DATASET_NAME):
        print("Loading data")
        with gzip.open(PROCESSED_DATASET_NAME, 'rt') as fin:
            data = json.load(fin)
            news_articles = data['news']
            print(f"{len(news_articles)} news articles")
            wikipedia = data['wikipedia']
            print(f"{len(wikipedia)} wikipedia pages")
            golden = data['golden']
            print(f"{len(golden)} manually generated paragraphs loaded")
    return data


def add_data_to_db(data: dict):
    # Given the {'news': [ articles }, 'wiki': [articles] } data, add all to the DB.
    db = sqlite3.open("data.db")
    db.row_factory = sqlite3.Row
    db.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY AUTO INCREMENT, source TEXT, text TEXT);")
    for source, doclist in data.items():
        for doc in doclist:
            db.execute("INSERT INTO documents (source, text) VALUES (?, ?);", (source, doc))
    db.commit()


def load_ground_truth():
    #ground_truth = dict()  # [Task][Dataset][Doc_IDX]
    with open(GROUND_TRUTH_DATASET_NAME, 'rt') as fin:
        ground_truth = json.load(fin)
        print(f"Loaded ground truth for tasks {ground_truth.keys()}")
        return ground_truth


def generate_data():
    wikipedia = generate_wikipedia()
    news_articles = generate_news()
    # Save All Datasets:
    data = {
        "wikipedia": wikipedia,
        "news": news_articles,
    }

    with gzip.open(PROCESSED_DATASET_NAME, 'wt') as fout:
        json.dump(data, fout)
    return data


def generate_wikipedia():
    # Wikipedia, extracting N articles at random.
    # Wikipedia was extracted from the RedPanda dataset, then sampled down to 0.1% of available articles for a total of 6552.
    # It was then further sampled as follows:
    with open("wikipedia_en_sampled.json", 'r') as fin:
        wikipedia = json.load(fin)
    wikipedia = [article for article in wikipedia if
                 "References" in article and len(article) > 250]
    # Cut off the 'References' and everything at the end because it tends to be noise.
    trimmed_wikipedia = list()
    for article in wikipedia:
        trimmed_wikipedia.append(article[:article.index("References")])
    wikipedia = trimmed_wikipedia
    return wikipedia


def generate_tweets():
    # Twitter, extract N tweets at random (omitting obscene entries).
    tweets = list()
    with open("tweets_sampled.csv", 'rt') as fin:
        cin = csv.reader(fin)
        header = next(cin)
        text_column = header.index("Tweet Content")
        rt_column = header.index("Tweet Type")
        for row in cin:
            if row[rt_column].lower() == "retweet":
                continue
            text = row[text_column]
            tweets.append(text)


def generate_news():
    # News Articles, extracting N articles.
    # %pip install unidecode

    csv.field_size_limit(1000000)

    csv_buffer = StringIO()
    with zipfile.ZipFile("/Users/josephcatrambone/Datasets/guardian_articles.zip",
                         'r') as fin:
        # print(fin.filelist)
        article_bytes = fin.read(name=fin.filelist[0].filename)
        csv_buffer.write(article_bytes.decode("utf-8"))
        csv_buffer.seek(0)
    cin = csv.reader(csv_buffer, )
    header = next(cin)
    body_index = header.index("bodyContent")
    print(f"Article content at index {body_index}")

    news_articles = list()
    for line in cin:
        article = unidecode(line[body_index])
        if len(article.strip()) > 100:
            news_articles.append(article)

    random.shuffle(news_articles)
    news_articles = news_articles[:MAX_SAMPLES_PER_SOURCE]
    return news_articles


def generate_bfcl():
    with open("/Users/josephcatrambone/Datasets/Berkeley-Function-Calling-Leaderboard/gorilla_openfunctions_v1_test_simple.json", 'rt') as fin:
        examples = [json.loads(line)["question"] for line in fin]
    return examples


def generate_amazon():
    # Pulls from the 2014 Amazon Reviews dataset and selects a small subset at random.
    reviews = list()
    with gzip.open("/Users/josephcatrambone/Datasets/amazon_review_dataset_2014_unduplicated_kcore_5.json.gz", 'rt') as fin:
        for line in fin:
            if random.random() < 0.001:
                reviews.append(json.loads(line)["reviewText"])
    return reviews
