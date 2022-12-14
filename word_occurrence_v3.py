
import collections
import re
import string
from typing import List
import glob

import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
import jieba


def main():
    """This function takes a corpus and prints:
    (a) word counts and (b) word counts by gender.
    """
    male_names = frozenset(['N', 'CG', 'Nick', 'R', 'Rick', 'R', 'Mark', 'Joe', 'Me', 'Ben', 'Ted', 'Me',
                            'Ronny', 'Gary', 'Bob', 'B', 'Brian', 'Wenchao', 'David', 'Scotty', 'Hal',
                            'Victor', 'Fernando', 'Horton', 'Tsung-han', 'Oshi', 'Alex', 'Kevin',
                            'Abram', 'shawn', 'TA', 'Ch', 'PB', 'Scotty', 'Bingbing'])
    female_names = frozenset(['Ashley', 'NC', 'Ella', 'Yu-tse', 'Mizu', 'MZ', 'GG', 'A'])
    all_tokens = []
    male_tokens = []
    female_tokens = []
    # Create a regular expression for direct quotes
    pattern = r"(^.+): (.+)"
    # Read all files and add them to all_tokens
    files = glob.glob("*.txt", recursive=True)
    for file in files:
        with open(file, "r") as source:
            for line in source:
                match = re.match(pattern, line)
                if match:
                    all_tokens.extend(parse_tokens(match.group(2)))
                    if match.group(1) in male_names:
                        male_tokens.extend(parse_tokens(match.group(2)))
                    elif match.group(1) in female_names:
                        female_tokens.extend(parse_tokens(match.group(2)))
    print(f"Overall word counts: ")
    for word, frequency in word_count(all_tokens):
        print(f'{word}: {frequency:,}')
    print()
    print(f"Male word counts: ")
    for word, frequency in word_count(male_tokens):
        print(f'{word}: {frequency:,}')
    print()
    print(f"Female word counts: ")
    for word, frequency in word_count(female_tokens):
        print(f'{word}: {frequency:,}')


def parse_tokens(quote: string) -> List:
    """This function parses English and Chinese tokens,
    and returns a list of parsed tokens.
    """
    parsed_quotes = []
    punc = string.punctuation + "’…-。...-"
    tokens = nltk.word_tokenize(quote)
    # print(tokens)
    # Parse Chinese words
    for token in tokens:
        if token not in punc:
            parsed_tokens = jieba.lcut(token)
            parsed_quotes.extend(parsed_tokens)
    return parsed_quotes


def word_count(parsed_tokens: List) -> List:
    """This function counts the frequencies of the tokens."""
    stop_words = frozenset(stopwords.words('english'))
    frequencies = collections.Counter()
    frequencies.update(token.casefold() for token in parsed_tokens if token not in stop_words)
    results = frequencies.most_common(15)
    return results


if __name__ == '__main__':
    main()
