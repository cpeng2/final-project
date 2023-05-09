
import collections
import re
import string
from typing import List
import glob

import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
import jieba
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from wordcloud import WordCloud

STOPWORDS = set(stopwords.words('english'))
FONTPATH = "PingFang.ttc"


def main():
    """This function takes a corpus and prints:
    (a) word counts and (b) word counts by gender.
    """
    male_names = frozenset(['N', 'CG', 'Nick', 'Me', 'R', 'Rick', 'Mark', 'Joe', 'Me', 'Ben', 'Ted',
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
                    name, quote = match.group(1), match.group(2)
                    all_tokens.extend(parse_tokens(quote))
                    if name in male_names:
                        male_tokens.extend(parse_tokens(quote))
                    elif name in female_names:
                        female_tokens.extend(parse_tokens(quote))

    # Print results
    print(f"There are {len(male_names)} male participants.")
    print(f"There are {len(female_names)} female participants.")
    print()
    print_results(all_tokens, "Overall word counts:")
    print_results(male_tokens, "Male word counts:")
    print_results(female_tokens, "Female word counts:")

    # Visualization
    # Specify the font
    font = fm.FontProperties(fname=FONTPATH)
    # Use the font in the plot
    plt.rcParams["font.family"] = font.get_name()
    # Draw plots
    plt.subplot(2, 1, 1)
    bar_plots(all_tokens, "Overall Frequencies")
    # male speakers
    plt.subplot(2, 2, 3)
    bar_plots(male_tokens, "Frequencies by male speakers")
    # female speakers
    plt.subplot(2, 2, 4)
    bar_plots(female_tokens, "Frequencies by female speakers")
    # plot charts
    plt.subplots_adjust(hspace=0.4)
    plt.show()
    # plt.savefig("frequency_results")
    plt.subplot(2, 1, 1)
    word_cloud(all_tokens, "Overall")
    plt.subplot(2, 2, 3)
    word_cloud(male_tokens, "Words by male speakers")
    plt.subplot(2, 2, 4)
    word_cloud(female_tokens, "Words by female speakers")
    plt.show()


def parse_tokens(quote: string) -> List:
    """This function parses English and Chinese tokens,
    and returns a list of parsed tokens.
    """
    parsed_quotes = []
    punc = string.punctuation + "’…-。...-，“”？"
    tokens = nltk.word_tokenize(quote)
    # Parse Chinese words
    for token in tokens:
        parsed_tokens = [word for word in jieba.lcut(token) if word not in punc]
        parsed_quotes.extend(parsed_tokens)
    return parsed_quotes


def word_count(parsed_tokens: List) -> List:
    """This function counts the frequencies of the tokens."""
    frequencies = collections.Counter()
    frequencies.update(token.casefold() for token in parsed_tokens if token not in STOPWORDS)
    results = frequencies.most_common(15)
    return results


def print_results(target_tokens, title) -> None:
    """This function takes a list and prints it."""
    print(f"{title}")
    for word, frequency in word_count(target_tokens):
        print(f'{word}: {frequency:,}')
    print()


def draw_charts(target_tokens, chart_name) -> None:
    """This function takes a list and makes the chart."""

    overall_count = [frequency for word, frequency in word_count(target_tokens)]
    bars = [word for word, frequency in word_count(target_tokens)]
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, overall_count, color=(0.2, 0.4, 0.6, 0.6))

    plt.title(f"{chart_name}")
    plt.xlabel("Word")
    plt.ylabel("Frequency")
    plt.xticks(y_pos, bars)


def bar_plots(tokens, chart_title) -> None:
    """This function takes a list and creates a bar chart."""
    words = [word for word, count in word_count(tokens)]
    counts = [count for word, count in word_count(tokens)]
    sns.barplot(x=words, y=counts)
    plt.title(f"{chart_title}")
    plt.xlabel("Words")
    plt.ylabel("Counts")


def word_cloud(tokens, title):
    text = " ".join(tokens)
    wordcloud = WordCloud(stopwords=STOPWORDS, font_path=FONTPATH, background_color="white").generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"{title}")
    # plt.savefig('wordcloud11.png')
    # plt.show()


if __name__ == '__main__':
    main()
