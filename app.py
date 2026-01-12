import streamlit as st
import pandas as pd
import plotly.express as px
import nltk
import random
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

st.set_page_config(page_title="Customer Feedback Analyzer", page_icon="💬", layout="wide")
st.title("Customer Feedback Sentiment Analyzer")
st.write("Generate highly random, fluent themed reviews or upload your own CSV.")

# --- Refresh button ---
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
# --- Refresh / Start Over Button ---
if st.button("🔄 Refresh / Start Over"):
    # Clear all session state
    st.session_state.clear()
    st.write("Refreshed! Reload the app or select options again.")  # optional message



# --- Review Generator ---
st.subheader("Generate Sample CSVs by Theme")
theme = st.selectbox("Choose a theme", ["Weather", "Service", "Feedback", "Reviews"])
num_reviews = st.slider("Number of reviews to generate", 5, 20, 5)

# Word pools for more randomness
subjects = ["I", "We", "My friend", "The team", "Customer service", "The product", "The staff", "Our experience"]
verbs_positive = ["loved", "enjoyed", "appreciated", "valued", "adored", "found excellent"]
verbs_negative = ["hated", "disliked", "was disappointed by", "was frustrated by", "couldn't enjoy"]
verbs_neutral = ["experienced", "noticed", "found", "tried", "observed", "encountered"]
objects_weather = ["the sunny day", "the rain", "stormy conditions", "the cold weather", "the heatwave"]
objects_service = ["the staff", "the support team", "the customer service", "the assistance", "the guidance"]
objects_feedback = ["the process", "the feedback system", "the form", "the survey", "the response"]
objects_reviews = ["the review process", "the opinions shared", "the ratings", "the comments", "the evaluations"]
adjectives = ["amazing", "terrible", "okay", "fantastic", "poor", "average", "excellent", "mediocre", "satisfying",
              "disappointing"]
contexts = ["during a rainy afternoon", "on a sunny morning", "after a long wait", "before the storm", "while deciding",
            "after reviewing all options"]


# Function to generate a single fluent review
def generate_fluent_review(theme_name):
    subj = random.choice(subjects)
    # pick verb and object based on theme
    if theme_name == "Weather":
        verb = random.choice(verbs_positive + verbs_negative + verbs_neutral)
        obj = random.choice(objects_weather)
    elif theme_name == "Service":
        verb = random.choice(verbs_positive + verbs_negative + verbs_neutral)
        obj = random.choice(objects_service)
    elif theme_name == "Feedback":
        verb = random.choice(verbs_positive + verbs_negative + verbs_neutral)
        obj = random.choice(objects_feedback)
    else:  # Reviews
        verb = random.choice(verbs_positive + verbs_negative + verbs_neutral)
        obj = random.choice(objects_reviews)

    adj = random.choice(adjectives)
    context = random.choice(contexts) if random.random() < 0.6 else ""

    # Build more fluent sentence with optional clauses
    review_templates = [
        f"{subj} {verb} {obj}. It was {adj} {context}.",
        f"Overall, {subj.lower()} {verb} {obj} and found it {adj}. {context}".strip(),
        f"{subj} felt that {obj} was {adj}. {context}",
        f"In my opinion, {obj} was {adj} {context}. {subj} {verb} it thoroughly.",
        f"{subj} would say the {obj} was {adj}. {context}."
    ]

    return random.choice(review_templates)


# Generate reviews button
if st.button(f"Generate {num_reviews} Themed Reviews CSV"):
    reviews = [generate_fluent_review(theme) for _ in range(num_reviews)]
    filename = f"sample_reviews_{theme.lower()}_{num_reviews}.csv"
    df_generated = pd.DataFrame({"review": reviews})
    df_generated.to_csv(filename, index=False)
    st.success(f"{filename} generated!")
    st.dataframe(df_generated)

# --- File Selection ---
st.subheader("Choose CSV for Analysis")
option = st.selectbox("Select CSV", ["Upload your own CSV", "Generated CSV"])
df = None
if option == "Upload your own CSV":
    uploaded_file = st.file_uploader("Drag and drop your CSV here", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
elif option == "Generated CSV":
    if 'df_generated' in locals():
        df = df_generated
    else:
        st.warning("Generate a CSV above first!")


# --- Sentiment Analysis ---
def analyze_sentiment(text):
    scores = sia.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment, compound


if df is not None and "review" in df.columns:
    df[["sentiment", "confidence"]] = df["review"].apply(lambda x: pd.Series(analyze_sentiment(str(x))))
    cols = [c for c in df.columns if c not in ["sentiment", "confidence"]] + ["sentiment", "confidence"]
    df = df[cols]

    # Stats
    total = len(df)
    pos = (df["sentiment"] == "Positive").sum()
    neg = (df["sentiment"] == "Negative").sum()
    neu = (df["sentiment"] == "Neutral").sum()
    st.subheader("Sentiment Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Reviews", total)
    c2.metric("Positive", pos)
    c3.metric("Negative", neg)
    c4.metric("Neutral", neu)

    # Bar Chart
    sentiment_counts = df["sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]
    sentiment_order = ["Positive", "Neutral", "Negative"]
    sentiment_counts["Sentiment"] = pd.Categorical(sentiment_counts["Sentiment"], categories=sentiment_order,
                                                   ordered=True)
    sentiment_counts = sentiment_counts.sort_values("Sentiment")
    fig = px.bar(
        sentiment_counts,
        x="Sentiment",
        y="Count",
        text="Count",
        color="Sentiment",
        color_discrete_map={"Positive": "#28a745", "Negative": "#dc3545", "Neutral": "#6c757d"},
        title="Customer Sentiment Distribution"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(template="plotly_white", title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)

    # Wide DataFrame
    st.subheader("Detailed Feedback Analysis")
    st.dataframe(df, use_container_width=True, height=500)
    st.success("Analysis complete!")
elif df is not None:
    st.error("CSV must contain a 'review' column.")
