"""
pipeline.py — Step 2: call the APIs once, cache what they say, flatten it into tables.

A pipeline built on someone else's API has two kinds of step, and keeping them
apart is the whole design:

- **Extract steps** call the API — slow, rate-limited, and the only part that can
  fail because of the network. Each one saves the raw JSON it got back to a cache
  file, and the next time it runs it reads that file instead of calling again.
  (Lesson 4-2: cache every call you can.)
- **Transform steps** turn that JSON into DataFrames. They never touch the
  network, so you can run them a hundred times while you get them right.

    reviews ─sentiment_step─▶ sentiment JSON ─sentences_table─▶ a row per sentence
                                            └opinions_table──▶ a row per target
    reviews ─entities_step──▶ entities JSON ─entities_table──▶ a row per entity
    mentions ─subjects_step─▶ a row per mention, with its subject

Each arrow changes the **grain** — review, sentence, target, entity — the idea
from Assignment 05, now coming out of JSON instead of a CSV.

Less scaffolding here: the steps are described, the code is yours.
"""

import json
import os

import pandas as pd

from .apicalls import classify_mentions, get_entities, get_sentiment
from .settings import MENTION_CATEGORIES


def sentiment_step(reviews: pd.DataFrame, cache_file: str) -> list:
    """Sentiment for every review, from the cache if it exists, else from the API.

    Returns a list with one dict per review: the Azure document
    (`response["results"]["documents"][0]`) with one extra key, `review_id`,
    copied from the review's row — so every document says which review it is.

    If `cache_file` exists it is loaded and returned with no API calls. Otherwise
    each review's `text` goes to `get_sentiment`, and the list is saved to
    `cache_file` before it is returned.
    """
    if os.path.exists(cache_file):
        with open(cache_file) as fh:
            return json.load(fh)
    documents = []
    for review in reviews.to_dict("records"):
        response = get_sentiment(review["text"])
        document = response["results"]["documents"][0]
        document["review_id"] = review["review_id"]
        documents.append(document)
    with open(cache_file, "w") as fh:
        json.dump(documents, fh)
    return documents


def entities_step(reviews: pd.DataFrame, cache_file: str) -> list:
    """Named entities for every review, cached — the same shape as `sentiment_step`.

    One dict per review: the Azure entity-recognition document with `review_id`
    added. Same cache rule, same loop, `get_entities` instead of `get_sentiment`.
    """
    if os.path.exists(cache_file):
        with open(cache_file) as fh:
            return json.load(fh)
    documents = []
    for review in reviews.to_dict("records"):
        response = get_entities(review["text"])
        document = response["results"]["documents"][0]
        document["review_id"] = review["review_id"]
        documents.append(document)
    with open(cache_file, "w") as fh:
        json.dump(documents, fh)
    return documents


SENTENCE_COLUMNS = ["review_id", "sentence_id", "sentence_text",
                    "sentence_sentiment", "sentence_offset", "sentence_length"]
OPINION_COLUMNS = ["review_id", "sentence_id", "mention", "mention_sentiment",
                   "assessments", "mention_offset", "source"]
ENTITY_COLUMNS = ["review_id", "entity_text", "entity_category", "entity_offset"]


def sentences_table(documents: list) -> pd.DataFrame:
    """One row per sentence: the sentiment documents flattened with `json_normalize`.

    Columns, in this order: `review_id`, `sentence_id`, `sentence_text`,
    `sentence_sentiment`, `sentence_offset`, `sentence_length`. `sentence_id` is
    the sentence's position in its review (0, 1, 2, ...) and the trailing space
    Azure leaves on `sentence_text` is stripped.
    """
    table = pd.json_normalize(documents, record_path="sentences",
                              meta=["review_id"], record_prefix="sentence_")
    if table.empty:
        return pd.DataFrame(columns=SENTENCE_COLUMNS)
    table["sentence_id"] = table.groupby("review_id").cumcount()
    table["sentence_text"] = table["sentence_text"].str.strip()
    return table[SENTENCE_COLUMNS].reset_index(drop=True)


def opinions_table(documents: list) -> pd.DataFrame:
    """One row per opinion target, with the words that carry the opinion.

    Columns, in this order: `review_id`, `sentence_id`, `mention`,
    `mention_sentiment`, `assessments`, `mention_offset`, `source`.

    A target points at its assessments through its `relations`; the number after
    `assessments/` in each `ref` is the assessment's position in the same
    sentence's `assessments` list. Only `"assessment"` relations count, and the
    texts are joined with `", "`. `source` is always `"opinion"`.
    """
    rows = []
    for document in documents:
        for sentence_id, sentence in enumerate(document["sentences"]):
            assessments = sentence.get("assessments", [])
            for target in sentence.get("targets", []):
                words = []
                for relation in target.get("relations", []):
                    if relation["relationType"] == "assessment":
                        position = int(relation["ref"].split("/")[-1])
                        words.append(assessments[position]["text"])
                rows.append({
                    "review_id": document["review_id"],
                    "sentence_id": sentence_id,
                    "mention": target["text"],
                    "mention_sentiment": target["sentiment"],
                    "assessments": ", ".join(words),
                    "mention_offset": target["offset"],
                    "source": "opinion",
                })
    return pd.DataFrame(rows, columns=OPINION_COLUMNS)


def entities_table(documents: list) -> pd.DataFrame:
    """One row per person or product the entity recognizer found.

    Columns, in this order: `review_id`, `entity_text`, `entity_category`,
    `entity_offset`. Only the categories in `MENTION_CATEGORIES` ("Person",
    "Product"); a plain 0, 1, 2, ... index.
    """
    table = pd.json_normalize(documents, record_path="entities",
                              meta=["review_id"], record_prefix="entity_")
    if table.empty:
        return pd.DataFrame(columns=ENTITY_COLUMNS)
    table = table[table["entity_category"].isin(MENTION_CATEGORIES)]
    return table[ENTITY_COLUMNS].reset_index(drop=True)


def subjects_step(reviews: pd.DataFrame, mentions: pd.DataFrame,
                  cache_file: str) -> pd.DataFrame:
    """What each mention is about — service, atmosphere, product, other — cached.

    Returns one row per (review, mention) with columns `review_id`, `mention`,
    `subject`, `subject_confidence`.

    Each review's mentions are sent to `classify_mentions` in `mention_offset`
    order, without repeats, so the same reviews always produce the same request
    (the portal only answers from its cache when a request is identical).
    """
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file)
    rows = []
    for review in reviews.to_dict("records"):
        own = mentions[mentions["review_id"] == review["review_id"]]
        if own.empty:
            continue
        names = (own.sort_values("mention_offset", kind="stable")["mention"]
                 .drop_duplicates().tolist())
        response = classify_mentions(review["text"], names)
        for name in names:
            answer = response["answers"][name]
            rows.append({
                "review_id": review["review_id"],
                "mention": name,
                "subject": answer["choice"],
                "subject_confidence": answer["confidence"],
            })
    subjects = pd.DataFrame(rows, columns=["review_id", "mention", "subject",
                                           "subject_confidence"])
    subjects.to_csv(cache_file, index=False)
    return subjects