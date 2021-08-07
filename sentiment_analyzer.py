import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.express as px
import seaborn as sns


class SentimentAnalyzer:
    def __init__(self, data_frame):
        self.df = data_frame
        self.sid = SentimentIntensityAnalyzer()

    def get_sentiment_df(self):
        sentiment_scores = self.df['Text'].apply(lambda x: self.sid.polarity_scores(x))
        df = pd.DataFrame(data=list(sentiment_scores))
        df['label'] = df.compound.apply(lambda x: 'neutral' if x == 0 else ('positive' if x > 0 else 'negative'))
        result = self.df.join(df.label)
        return result

    def show_sentiment_df(self):
        df = self.get_sentiment_df()
        counts_df = df.label.value_counts().reset_index()
        px.bar(counts_df, x='index', y='label').show()
