
import re

import glob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# nltk.download('vader_lexicon')


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
                    p_score = sid.polarity_scores(match.group(2))
                    all_scores.append(p_score["compound"])
                    if match.group(1) in male_names:
                        p_score = sid.polarity_scores(match.group(2))
                        male_scores.append(p_score["compound"])
                    elif match.group(1) in female_names:
                        p_score = sid.polarity_scores(match.group(2))
                        female_scores.append(p_score["compound"])
    print(sum(all_scores)/len(all_scores))
    print(sum(male_scores)/len(male_scores))
    print(sum(female_scores)/len(female_scores))

    """The Compound score is a metric that calculates the sum of all the lexicon ratings 
    which have been normalized between -1(most extreme negative) and +1 (most extreme positive).
    positive sentiment: (compound score >= 0.05) 
    neutral sentiment: (compound score > -0.05) and (compound score < 0.05)
    negative sentiment: (compound score <= -0.05)"""


if __name__ == '__main__':
    main()
