
import re
import glob
import string
from typing import Dict, List

import jieba
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy

# Import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import VotingClassifier


def extract_features(sentence: str):
    features: Dict[str, bool] = {}
    tokens = parse_tokens(sentence)
    gen_dist_words = frozenset([
        # male words
        "people", "get", "chinese", "like", "get", "right", "oh"
        # female words
        "我", "是", "有", "在", "人", "know", "china"])
    sid = SentimentIntensityAnalyzer()
    # Feature 1: gender-distinguishing words
    for token in tokens:
        if token in gen_dist_words:
            features[f"has.{token}"] = True
    # Feature 2: code-switching
    cs = count_code_switching(sentence)
    if cs > 0:
        features["codeswitch"] = True
    # Feature 3: target language use
    target_lang = target_lang_use(sentence)
    if target_lang > 0:
        features["TL"] = True
    # Feature 4: polarity score
    p_score = sid.polarity_scores(sentence)
    if p_score["compound"] >= .10:
        features['positive'] = True
    return features


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


def print_misclassified(model: str, y_test: List, y_pred: List, sentences: List):
    """THis function print misclassified sentences to a csv file."""
    with open("misclassified.csv", "a") as sink:
        misclassified_indices = [i for i in range(len(y_test)) if y_test[i] != y_pred[i]]
        print(f"{model}Misclassified sentences:", file=sink)
        for i in misclassified_indices:
            print(y_test[i], y_pred[i], sentences[i], file=sink)


def main():
    """This function takes a corpus and prints:
    (a) word counts and (b) word counts by gender.
    """
    male_names = frozenset(['N', 'CG', 'Nick', 'Me', 'R', 'Rick', 'Mark', 'Joe', 'Me', 'Ben', 'Ted',
                            'Ronny', 'Gary', 'Bob', 'B', 'Brian', 'Wenchao', 'David', 'Scotty', 'Hal',
                            'Victor', 'Fernando', 'Horton', 'Tsung-han', 'Oshi', 'Alex', 'Kevin',
                            'Abram', 'shawn', 'TA', 'Ch', 'PB', 'Scotty', 'Bingbing'])
    female_names = frozenset(['Ashley', 'NC', 'Ella', 'Yu-tse', 'Mizu', 'MZ', 'GG', 'A'])
    male_sent = []
    female_sent = []
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
                    if name in male_names:
                        male_sent.append(quote)
                    elif name in female_names:
                        female_sent.append(quote)

    # Prepare the data
    sentences = male_sent + female_sent  # List of sentences
    print(len(sentences))
    # List of corresponding labels (male or female)
    labels = ["male" for _ in male_sent] + ["female" for _ in female_sent]

    # Extract features: TF-IDF
    vectorizer = TfidfVectorizer(min_df=3, max_df=.9)
    D = vectorizer.fit_transform(sentences)
    print(D.size)
    # Split the data (train 0.8, test 0.1, dev 0.1)
    seed = 11
    D_train, D_other, y_train, y_other = train_test_split(
        D, labels, test_size=.2, random_state=seed)
    D_test, D_dev, y_test, y_dev = train_test_split(
        D_other, y_other, test_size=.5, random_state=seed)
    print(f"train size: {D_train.size}, dev size: "
          f"{D_dev.size}, test size: {D_test.size}")

    # Train the dummy classifier (baseline)
    clf0 = DummyClassifier()
    clf0.fit(D_train, y_train)
    # Evaluate
    y_pred = clf0.predict(D_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Testing baseline accuracy:\t{accuracy:.4f}")
    print_misclassified("Dummy classifier", y_test, y_pred, sentences)

    # Train the logistic regression classifier
    clf1 = LogisticRegression(C=5)
    clf1.fit(D_train, y_train)
    # Tuning
    model = LogisticRegression(solver="saga", penalty="elasticnet")
    grid = {"C": numpy.array([.1, .2, .5, 1., 2., 5., 10., 20., 50.]),
            "l1_ratio": numpy.array([.1, .5, .9])}
    search = GridSearchCV(model, grid, n_jobs=-1)
    search.fit(D_dev, y_dev)
    print(search.best_params_)
    # Evaluate
    y_pred = clf1.predict(D_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Testing logistic regression accuracy:\t{accuracy:.4f}")
    print_misclassified("Logistic regression", y_test, y_pred, sentences)

    # Train the naive Bayes classifier
    clf2 = BernoulliNB(alpha=1)
    clf2.fit(D_train, y_train)
    # Evaluate
    y_pred = clf2.predict(D_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Testing Naive Bayes accuracy:\t{accuracy:.4f}")
    print_misclassified("Naive Bayes", y_test, y_pred, sentences)

    # voting classifier
    eclf = VotingClassifier(estimators=[
        ('Dummy classifier', clf0),
        ('Logistic regression classifier', clf1),
        ('Naive Bayes classifier', clf2)],
        voting="hard")
    eclf = eclf.fit(D_train, y_train)
    # print(eclf1.predict(D_test))
    for classifier, label in zip([clf0, clf1, clf2, eclf],
                                ["Dummy", "Logistic Regression", "Naive Bayes", "Ensamble"]):
        scores = cross_val_score(classifier, D_test, y_test, scoring="accuracy", cv=10)
        print("Accuracy: %0.4f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))


if __name__ == '__main__':
    main()