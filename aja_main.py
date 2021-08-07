import string

import pandas as pd
import plotly.express as px
import seaborn as sns
from nltk.corpus import stopwords
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('stopwords')
nltk.download('vader_lexicon')
from collections import Counter

sns.set(style="darkgrid")

# READ CSV
df = pd.read_csv('https://raw.githubusercontent.com/gabrielpreda/covid-19-tweets/master/covid19_tweets.csv')
df.head()

# get data shape = df.shape

# SELECT NEEDED COLUMNS
needed_columns = ['user_name', 'date', 'text']
df = df[needed_columns]
df.head()

# Assign categories to columns
df.user_name = df.user_name.astype('category')
df.user_name = df.user_name.cat.codes
df.date = pd.to_datetime(df.date).dt.date
df.head()

# See tweets
texts = df['text']
# texts

# remove urls
remove_url = lambda x: re.sub(r'https\S+', '', str(x))
texts_lr = texts.apply(remove_url)
# texts_lr

# everything to minus
to_lower = lambda x: x.lower()
texts_lr_lc = texts_lr.apply(to_lower)
# texts_lr_lc

# remove punctuation
remove_puncs = lambda x: x.translate(str.maketrans('', '', string.punctuation))
text_lr_lc_np = texts_lr_lc.apply(remove_puncs)
# text_lr_lc_np

# remove stop words
more_words = ['covid', '#coronavirus', '#coronavirusoutbreak', '#coronavirusPandemic', '#covid19', '#covid_19',
              '#epitwitter', '#ihavecorona', 'amp', 'coronavirus', 'covid19']
stop_words = set(stopwords.words('english'))
stop_words.update(more_words)
remove_words = lambda x: ' '.join([word for word in x.split() if word not in stop_words])
text_lr_lc_np_ns = text_lr_lc_np.apply(remove_words)
# text_lr_lc_np_ns

# Most used words
words_list = [word for line in text_lr_lc_np_ns for word in line.split()]
word_counts = Counter(words_list).most_common(50)
# plot them
words_df = pd.DataFrame(word_counts)
words_df.columns = ['word', 'frequency']
px.bar(words_df, x='word', y='frequency', title='Most common words')

# cleaned text in the main dataframe
df.text = text_lr_lc_np_ns
df.head()

# Analyze sentiments
sid = SentimentIntensityAnalyzer()
ps = lambda x: sid.polarity_scores(x)
sentiment_scores = df.text.apply(ps)
# sentiment_scores
# show table
sentiment_df = pd.DataFrame(data=list(sentiment_scores))
sentiment_df.head()

# put labels on the table based of sentiment
labelize = lambda x: 'neutral' if x == 0 else ('positive' if x > 0 else 'negative')
sentiment_df['label'] = sentiment_df.compound.apply(labelize)
sentiment_df.head()

# Join the sentiments with the actual data
data = df.join(sentiment_df.label)
data.head()

# show how many entries there are with each label
counts_df = data.label.value_counts().reset_index()
# counts_df

# barplot it
sns.barplot(x='index', y='label', data=counts_df)

data.head()

# Plot the evolution through time
data_agg = data[['user_name', 'date', 'label']].groupby(['date', 'label']).count().reset_index()
data_agg.columns = ['date', 'label', 'counts']
data_agg.head()

px.line(data_agg, x='date', y='counts', color='label', title='daily tweets sentimental analysis')
