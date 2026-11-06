"""
review_explorer.py — what Salt City Coffee's customers are saying, at a glance.

The owner reads every review on the shop's website, but thirty reviews across
three locations is more than anyone can hold in their head. This page shows
what customers praise and complain about — by location and by subject — and
answers plain-English questions like "Do people like the mocha latte?" or "What
is the most popular product?" from the reviews themselves, with the evidence
underneath.

The page is assembly again: `analyze_reviews` does all the API work (from the
cache after the first run), and the Q&A box is three calls you have already
written — find the question's scope and evidence, build the messages, ask the
model.

Run it:  Run and Debug -> "Streamlit Run: Current File"
Test it: pytest tests/test_pipeline.py -k app
"""

import os

import streamlit as st

from reviews import (
    REVIEWS_FILE,
    analyze_reviews,
    answer_question,
    ask_llm,
    build_messages,
    describe_scope,
    load_aliases,
    summarize_items,
)

# 1. Which reviews
reviews_file = os.environ.get("REVIEWS_FILE", REVIEWS_FILE)

# 2. Heading
st.title("Salt City Coffee — What Customers Are Saying")
st.markdown(
    "Every customer review, broken into the things people talk about. Filter by "
    "location or subject, or ask a question at the bottom of the page."
)

# 3. The data
mentions = analyze_reviews(reviews_file)

# 4. Two filters, side by side
location_column, subject_column = st.columns(2)
location = location_column.selectbox(
    "Location", ["All"] + sorted(mentions["location"].unique()), key="location")
subject = subject_column.selectbox(
    "Subject", ["All"] + sorted(mentions["subject"].unique()), key="subject")

selected = mentions
if location != "All":
    selected = selected[selected["location"] == location]
if subject != "All":
    selected = selected[selected["subject"] == subject]

# 5. Four metrics, side by side
sentiment = selected["mention_sentiment"]
reviews_col, mentions_col, positive_col, negative_col = st.columns(4)
reviews_col.metric("Reviews", int(selected["review_id"].nunique()))
mentions_col.metric("Mentions", len(selected))
positive_col.metric("Positive", int((sentiment == "positive").sum()))
negative_col.metric("Negative", int((sentiment == "negative").sum()))

# 6. Three tables in three tabs
summary = summarize_items(selected)
praised_tab, complained_tab, said_tab = st.tabs(
    ["Most praised", "Most complained about", "What people said"])
praised = summary[summary["net"] > 0].sort_values(
    "net", ascending=False, kind="stable")
praised_tab.dataframe(praised.head(5), hide_index=True)
complained = summary[summary["net"] < 0].sort_values(
    "net", ascending=True, kind="stable")
complained_tab.dataframe(complained.head(5), hide_index=True)
said_tab.dataframe(
    selected[["date", "location", "item", "subject", "mention_sentiment",
              "assessments", "sentence_text"]],
    hide_index=True)

# 7. The question box
st.subheader("Ask the reviews")
question = st.text_input("Your question", key="question",
                         placeholder="What is the most popular product?")
asked = st.button("Ask", key="ask")

# 8. When Ask is clicked
if asked and question.strip():
    scope, evidence = answer_question(question, mentions, load_aliases())
    if evidence.empty:
        st.warning(
            f"No reviews mention {describe_scope(scope)}. Try a menu item, a "
            "staff member, or part of the shop."
        )
    else:
        messages = build_messages(question, evidence)
        with st.spinner("Asking the model..."):
            answer = ask_llm(messages)
        st.markdown(answer)
        st.caption(
            f"Based on {len(evidence)} mentions — {describe_scope(scope)}:")
        st.dataframe(
            evidence[["date", "location", "item", "mention_sentiment",
                      "sentence_text"]],
            hide_index=True)