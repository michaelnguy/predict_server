from flask import Flask, request, jsonify
import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import emoji
import contractions
from io import BytesIO
from PIL import Image
from collections import OrderedDict
import requests
import torch
import sys
import os

sys.path.append(os.path.dirname(__file__))
# import json
from transformers import AutoImageProcessor, ViTModel

# from data_utils import load_image_tags
# from index import create_index_from_df
# from tag_recommender import TagRecommender

# Constants (these can be parameterized if needed)
# DATA_DIR = "./HARRISON/"
# DATA_LEN = 57383
# INDICES_SPLIT_FILE_NAME = "data_indices_split.json"
# IMG_PATHS_FILE_NAME = "data_list.txt"
# GT_TAGS_FILE_NAME = "tag_list.txt"
# EMBEDDINGS_FILE_NAME = "data_embeddings.txt"
# ID_COL = "img_id"
# EMB_COL = "emb"


# def load_data():
#     # Initialize an empty dataframe
#     emb_df = pd.DataFrame()

#     # List of your 5 .txt files
#     embedding_files = [
#         os.path.join(DATA_DIR, "output_part_1.txt"),
#         os.path.join(DATA_DIR, "output_part_2.txt"),
#         os.path.join(DATA_DIR, "output_part_3.txt"),
#         os.path.join(DATA_DIR, "output_part_4.txt"),
#         os.path.join(DATA_DIR, "output_part_5.txt"),
#     ]

# Load and concatenate all embedding files
# for file in embedding_files:
#     temp_df = pd.read_json(file, lines=True)
#     emb_df = pd.concat([emb_df, temp_df], ignore_index=True)
# with open(os.path.join(DATA_DIR, INDICES_SPLIT_FILE_NAME)) as f:
#     indices_split = json.load(f)

# train_df = emb_df[emb_df[ID_COL].isin(indices_split["train"])]
# val_df = emb_df[emb_df[ID_COL].isin(indices_split["val"])]
# val_idx_mapping = {i: img_id for i, img_id in enumerate(val_df[ID_COL])}

# return train_df, val_df, val_idx_mapping


# def create_recommender(train_df):
#     img_index, train_idx_mapping = create_index_from_df(train_df)
#     tags_list = load_image_tags(os.path.join(DATA_DIR, GT_TAGS_FILE_NAME))
#     recommender = TagRecommender(img_index, train_idx_mapping, tags_list)

#     return recommender


IMG_MODEL_NAME = "google/vit-base-patch16-224-in21k"

VIT_IMAGE_PROCESSOR = AutoImageProcessor.from_pretrained(IMG_MODEL_NAME)
VIT_MODEL = ViTModel.from_pretrained(IMG_MODEL_NAME)


def get_image_from_url(img_url):
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content))
    return img


def get_embeddings_from_model(images, image_processor, model):
    inputs = image_processor(images, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state
    embeddings = last_hidden_states[:, 0].cpu()
    return embeddings


def img_vectorizer(images):
    embeddings = get_embeddings_from_model(images, VIT_IMAGE_PROCESSOR, VIT_MODEL)
    return embeddings.numpy()


# Load the vectorizer and model
with open("vectorizer.pickle", "rb") as f:
    vectoriser = pickle.load(f)

with open("Sentiment-LR.pickle", "rb") as f:
    model = pickle.load(f)

nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("punkt")
nltk.download("stopwords")

# Initialize the necessary tools
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


def expand_contractions(text):
    return contractions.fix(text)


app = Flask(__name__)

# Load the data and create the recommender once when the server starts
# train_df, val_df, val_idx_mapping = load_data()
# recommender = create_recommender(train_df)

# with open("recommender.pkl", "wb") as f:
#     pickle.dump(recommender, f)

with open("recommender.pkl", "rb") as f:
    recommender_test = pickle.load(f)


@app.route("/get_hashtags", methods=["POST"])
def get_hashtags():
    try:
        data = request.json
        image_url = data["image_url"]

        # Process the image and get hashtags
        image = get_image_from_url(image_url)
        images = [image]
        pred_tags = recommender_test.get_tags_for_image(images, img_vectorizer)

        return jsonify({"image_url": image_url, "tags": pred_tags})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Expand contractions
    text = expand_contractions(text)
    # Convert emojis to text
    text = emoji.demojize(text)
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text, flags=re.MULTILINE)
    # Remove mentions and hashtags
    text = re.sub(r"@\w+|#\w+", " ", text)
    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    # Tokenize text
    tokens = nltk.word_tokenize(text)
    # Lemmatize and stem each token
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token is not None]
    tokens = [stemmer.stem(token) for token in tokens]
    # Remove stop words
    tokens = [token for token in tokens if token not in stop_words]
    # Remove duplicate tokens while preserving order
    tokens = list(OrderedDict.fromkeys(tokens))
    # Remove extra whitespace
    text = " ".join(tokens)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        texts = data["texts"]

        # Predict the sentiment
        preprocessed_texts = [preprocess_text(text) for text in texts]
        textdata = vectoriser.transform(preprocessed_texts)
        sentiment = model.predict(textdata)

        # Make a list of text with sentiment.
        results = []
        for text, pred in zip(texts, sentiment):
            # Look for positive emojis and hard code sentiment

            results.append({"text": text, "sentiment": int(pred)})

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
