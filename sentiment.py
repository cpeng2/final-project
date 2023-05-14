
import re

import glob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# nltk.download('vader_lexicon')
from nltk.corpus import stopwords
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


def main():
    """This function takes a corpus and prints:
    (a) word counts and (b) word counts by gender.
    """
    male_names = frozenset(['N', 'CG', 'Nick', 'R', 'Rick', 'R', 'Mark', 'Joe', 'Me', 'Ben', 'Ted', 'Me',
                            'Ronny', 'Gary', 'Bob', 'B', 'Brian', 'Wenchao', 'David', 'Scotty', 'Hal',
                            'Victor', 'Fernando', 'Horton', 'Tsung-han', 'Oshi', 'Alex', 'Kevin',
                            'Abram', 'shawn', 'TA', 'Ch', 'PB', 'Scotty', 'Bingbing'])
    female_names = frozenset(['Ashley', 'NC', 'Ella', 'Yu-tse', 'Mizu', 'MZ', 'GG', 'A'])
    all_scores = []
    male_scores = []
    female_scores = []
    sid = SentimentIntensityAnalyzer()
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
                    p_score = sid.polarity_scores(quote)
                    all_scores.append(p_score["compound"])
                    if name in male_names:
                        p_score = sid.polarity_scores(quote)
                        male_scores.append(p_score["compound"])
                    elif name in female_names:
                        p_score = sid.polarity_scores(quote)
                        female_scores.append(p_score["compound"])
    overall_sent = sum(all_scores)/len(all_scores)
    male_sent = sum(male_scores)/len(male_scores)
    female_sent = sum(female_scores)/len(female_scores)
    print(f"Overall: {overall_sent:.2f}, Male: {male_sent:.2f}, Female: {female_sent:.2f}")
    t_test(male_scores, female_scores)

    # visualization
    sents = [overall_sent, male_sent, female_sent]
    bars = ("Overall", "Male", "Female")
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, sents, color=(0.2, 0.4, 0.6, 0.6))
    plt.title("Sentiment Analysis")
    plt.xlabel("Gender")
    plt.ylabel("Sentiment Intensity")
    plt.xticks(y_pos, bars)
    plt.show()

    """The Compound score is a metric that calculates the sum of all the lexicon ratings 
    which have been normalized between -1(most extreme negative) and +1 (most extreme positive).
    positive sentiment: (compound score >= 0.05) 
    neutral sentiment: (compound score > -0.05) and (compound score < 0.05)
    negative sentiment: (compound score <= -0.05)"""


def t_test(tokens1, tokens2):
    sample1 = np.array(tokens1)
    sample2 = np.array(tokens2)
    t_stat, p_val = ttest_ind(sample1, sample2)
    # print("t-statistic:", t_stat)
    print(f"p-value= {p_val:.2f}")


if __name__ == '__main__':
    main()
