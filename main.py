import os
from pprint import pprint

from PyInquirer import prompt
from prompt_toolkit.validation import Validator, ValidationError

from data_analyzer import DataAnalyzer
from sentiment_analyzer import SentimentAnalyzer


class FileValidator(Validator):
    def validate(self, document):
        path = document.text.strip()
        file_exists = os.path.isfile(path)
        if not file_exists or not path.split('.')[-1] == 'csv':
            raise ValidationError(
                message='Please enter a valid csv file path',
                cursor_position=len(document.text))


def main():
    questions = [
        {
            'type': 'input',
            'name': 'file_location',
            'message': 'Input tweets .csv file location',
            'validate': FileValidator
        },
        # {
        #     'type': 'list',
        #     'name': 'selected_analysis',
        # }
    ]
    answers = prompt(questions)
    data_analyzer = DataAnalyzer(answers['file_location'].strip(), 'spanish')
    data_analyzer.show_most_used_words(50).show()

    sentiment_analyzer = SentimentAnalyzer(data_analyzer.get_data_frame())
    sentiment_analyzer.show_sentiment_df()


if __name__ == '__main__':
    main()


