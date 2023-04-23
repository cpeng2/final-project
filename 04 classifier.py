
import re
import glob


# Import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB


def main():
    """This function takes a corpus and prints:
    (a) word counts and (b) word counts by gender.
    """
    male_names = frozenset(['N', 'CG', 'Nick', 'Me', 'R', 'Rick', 'Mark', 'Joe', 'Me', 'Ben', 'Ted',
                            'Ronny', 'Gary', 'Bob', 'B', 'Brian', 'Wenchao', 'David', 'Scotty', 'Hal',
                            'Victor', 'Fernando', 'Horton', 'Tsung-han', 'Oshi', 'Alex', 'Kevin',
                            'Abram', 'shawn', 'TA', 'Ch', 'PB', 'Scotty', 'Bingbing'])
    female_names = frozenset(['Ashley', 'NC', 'Ella', 'Yu-tse', 'Mizu', 'MZ', 'GG', 'A'])
    # l1_speakers = frozenset(["CG"])
    # l2_speakers = frozenset(["N", "Nick", "Me"])
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
                    if match.group(1) in male_names:
                        male_sent.append(match.group(2))
                    elif match.group(1) in female_names:
                        female_sent.append(match.group(2))

    # Prepare the data
    sentences = male_sent + female_sent  # List of sentences
    # List of corresponding labels (male or female)
    labels = ["male" for _ in male_sent] + ["female" for _ in female_sent]
    # Extract features
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sentences)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=11)

    # Train the logistic regression classifier
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    # Evaluate the classifier
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Logistic regression accuracy: {accuracy:.2f}")

    # Train the naive bayes classifier
    classifier = BernoulliNB(alpha=1)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Naive Bayes accuracy: {accuracy:.2f}")


if __name__ == '__main__':
    main()
