
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
import seaborn as sns
from scipy.stats import ttest_ind

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
    male_codeswitch = []
    female_codeswitch = []
    male_target_lang = []
    female_target_lang = []
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
                        male_codeswitch.append(count_code_switching(quote))
                        male_target_lang.append(target_lang_use(quote))
                    elif name in female_names:
                        female_tokens.extend(parse_tokens(quote))
                        female_codeswitch.append(count_code_switching(quote))
                        female_target_lang.append(target_lang_use(quote))

    # Print results
    print(f"There are {len(male_names)} male participants.")
    print(f"There are {len(female_names)} female participants.")
    print()
    print("Code-switching results:")
    m_co_ratio = sum(male_codeswitch) / len(male_codeswitch)
    f_co_ratio = sum(female_codeswitch) / len(female_codeswitch)
    print(f"Male speakers code-switched {sum(male_codeswitch)} times,"
          f" or {m_co_ratio * 100} times per 100 sentence.")
    print(f"Female speakers code-switched {sum(female_codeswitch)} times,"
          f" or {f_co_ratio * 100} times per 100 sentence.")
    t_test(male_codeswitch, female_codeswitch)
    print()
    print("Target language use results:")
    male_target_lang_ratio = sum(male_target_lang) / len(male_target_lang) * 100
    female_target_lang_ratio = sum(female_target_lang) / len(female_target_lang) * 100
    print(f"Male use of target language: {male_target_lang_ratio:.2f}%")
    print(f"Female use of target language: {female_target_lang_ratio:.2f}%")
    t_test(male_target_lang, female_target_lang)

    # visualization
    codeswitch = [m_co_ratio * 100, f_co_ratio * 100]
    bars = ("Male", "Female")
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, codeswitch, color=(0.2, 0.4, 0.6))
    plt.title("Code-switching")
    plt.xlabel("Gender")
    plt.ylabel("Times of Code-switching per 100 sentences")
    plt.xticks(y_pos, bars)
    plt.show()

    target_lang = [male_target_lang_ratio, female_target_lang_ratio]
    bars = ("Male", "Female")
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, target_lang, color=(0.2, 0.6, 0.6))
    plt.title("Target Language Use")
    plt.xlabel("Gender")
    plt.ylabel("Percentage of target language use")
    plt.xticks(y_pos, bars)
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


def count_code_switching(sentence: str):
    tokens = nltk.word_tokenize(sentence)
    no_punc = [token for token in tokens
               if token not in string.punctuation + "’…-。...-，“”？"]
    code_switch_count = 0
    for i in range(len(no_punc) - 1):
        (w1, w2) = (no_punc[i], no_punc[i + 1])
        if w1.isascii() and not w2.isascii():
            code_switch_count += 1
        elif not w1.isascii() and w2.isascii():
            code_switch_count += 1
    return code_switch_count


def target_lang_use(sentence: str):
    tokens = parse_tokens(sentence)
    no_punc = [token for token in tokens
               if token not in string.punctuation + "’…-。...-，“”？"]
    en = 0
    zh = 0
    for token in no_punc:
        if token.isascii():
            en += 1
        else:
            zh += 1
    return zh / (zh + en)


def t_test(tokens1, tokens2):
    sample1 = np.array(tokens1)
    sample2 = np.array(tokens2)
    t_stat, p_val = ttest_ind(sample1, sample2)
    # print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_val:.4f}")


if __name__ == '__main__':
    main()
