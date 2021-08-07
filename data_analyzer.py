import re
import string
import pandas as pd
from nltk.corpus import stopwords
from collections import Counter
import plotly.express as px
from text_processor import TextProcessor


class DataAnalyzer:
    needed_columns = ['Name', 'Date', 'Text']
    words_to_remove = ['covid', '#coronavirus', '#coronavirusoutbreak', '#coronavirusPandemic', '#covid19', '#covid_19',
                       '#epitwitter', '#ihavecorona', 'amp', 'coronavirus', 'covid19']

    def __init__(self, csv_file_path: string, language: string):
        self.data_frame = pd.read_csv(csv_file_path, encoding="ISO-8859-1")
        self.data_frame = self.data_frame[self.needed_columns]
        self.__assign_types()
        self.dirty_data_frame = self.data_frame
        self.clean_tweets = TextProcessor(self.data_frame['Text'], language).clean_text(self.words_to_remove)
        self.data_frame['Text'] = self.clean_tweets

    def __assign_types(self):
        self.data_frame.Name = self.data_frame.Name.astype('category')
        self.data_frame.Name = self.data_frame.Name.cat.codes
        self.data_frame.Date = pd.to_datetime(self.data_frame.Date).dt.date

    def get_tweets(self):
        return self.data_frame['Text']

    def get_data_frame(self):
        return self.data_frame

    def get_dirty_data_frame(self):
        return self.dirty_data_frame

    def __count_most_used_words(self, max_words=50):
        words_list = [word for line in self.clean_tweets for word in line.split()]
        return Counter(words_list).most_common(max_words)

    def show_most_used_words(self, max_words=50):
        words_df = pd.DataFrame(self.__count_most_used_words(max_words))
        words_df.columns = ['word', 'frequency']
        return px.bar(words_df, x='word', y='frequency', title='Most common words')
