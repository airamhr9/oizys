import re
import string

from nltk.corpus import stopwords


class TextProcessor:
    def __init__(self, text, language: string):
        self.text = text
        self.language = language.lower()

    def clean_text(self, words_to_remove):
        self.text = self.text.apply(lambda x: remove_urls(x))
        self.text = self.text.apply(lambda x: remove_punctuation(x))
        self.text = self.text.apply(lambda x: x.lower())
        self.text = self.text.apply(lambda x: remove_words(x, self.language, words_to_remove))
        return self.text


def remove_urls(text):
    return re.sub(r'https\S+', '', str(text))


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_words(text, language, words_to_remove):
    no_value_words = set(stopwords.words(language))
    no_value_words_english = set(stopwords.words('english'))
    no_value_words.update(words_to_remove)
    no_value_words.update(no_value_words_english)
    no_value_words.update(['rt'])
    return ' '.join([word for word in text.split() if word not in no_value_words])
