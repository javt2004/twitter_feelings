import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

df = pd.read_csv("twitter_feelings/csv/translated.csv", encoding="utf-8")

char_map = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ü": "u", "ñ": "n"}

stop_words = set(stopwords.words("spanish"))


def delete_mentions(text):
    return re.sub(r"@\w+", "", text)


def delete_urls(text):
    return re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)


def delete_no_ascii(text):
    return re.sub(r"[^\x00-\x7FñáéíóúüÁÉÍÓÚÜ]", "", text)


def replace_special_chars(text):
    for char, repl in char_map.items():
        text = text.replace(char, repl)
    return text


def delete_numbers(text):
    return re.sub(r"\d+", "", text)


def delete_punctuation(text):
    # Eliminar todos los signos de puntuación excepto !, ?, y ...
    text = re.sub(r"(?<!\.)\.(?!\.)", "", text)
    return re.sub(r"[^\w\s!?…]", "", text)


def delete_repeated_characters(text):
    return re.sub(r"(.)\1+", r"\1\1", text)


def convert_to_lowercase(text):
    return text.lower()


def delete_stop_words(text):
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)


def clean_data(df):
    if "tweet_id" in df.columns:
        df = df.drop(columns=["tweet_id"])

    df["content"] = df["content"].apply(delete_mentions)
    df["content"] = df["content"].apply(delete_urls)
    df["content"] = df["content"].apply(delete_no_ascii)
    df["content"] = df["content"].apply(replace_special_chars)
    df["content"] = df["content"].apply(delete_numbers)
    # df["content"] = df["content"].apply(delete_punctuation)
    df["content"] = df["content"].apply(delete_repeated_characters)
    # df["content"] = df["content"].apply(convert_to_lowercase)
    df["content"] = df["content"].apply(delete_stop_words)

    df.to_csv("twitter_feelings/csv/cleaned.csv", index=False)


clean_data(df)
