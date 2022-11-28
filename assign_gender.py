
import re
import jieba
import collections
import string
import nltk
assert nltk.download('punkt')


# Make a list of the target text files
file1 = '0430_fieldnotes.txt'
file2 = '0408_fieldnotes.txt'
file3 = '0412_fieldnotes.txt'
file4 = '0430_fieldnotes.txt'
file5 = '0607_fieldnotes.txt'
files = [file1, file2, file3, file4, file5]


def main():
    # Create a regular expression for direct quotes
    pattern = r"^.+:"
    # Read all files and add them to male_words and female_words
    male_words = []
    female_words = []
    for file in files:
        with open(file, 'r') as source:
            for line in source:
                if re.search(pattern, line, re.IGNORECASE):
                    tokens = nltk.word_tokenize(line)
                    # Parse Chinese words
                    parsed_line = []
                    for token in tokens:
                        parsed_tokens = jieba.lcut(token)
                        parsed_line += parsed_tokens
                        """When I print out parsed_line here, 
                        some of the sentences will be printed multiple times.
                        Not sure why."""
                        # Assign words to male_words or female_words
                        for token in parsed_line:
                            if assign_gender(parsed_line) == 1:
                                male_words.append(token)
                            elif assign_gender(parsed_line) == 2:
                                female_words.append(token)
    print(f'Word counts for male speakers:')
    count(male_words)
    print()
    print(f'Word counts for female speakers:')
    count(female_words)


def count(all_tokens):
    punctuation = frozenset(string.punctuation)
    frequencies = collections.Counter()
    frequencies.update(token.casefold() for token in all_tokens if token not in punctuation)
    for word, count in sorted(frequencies.items(), key=lambda t: t[1], reverse=False):
        print(f'{word}: {count:,}')


def assign_gender(tokens):
    male_names = frozenset(['N', 'CG', 'Nick', 'R', 'Rick', 'R', 'Mark', 'Joe', 'Me', 'Ben', 'Ted', 'Me',
                            'Ronny', 'Gary', 'Bob', 'B', 'Brian', 'Wenchao', 'David', 'Scotty', 'Hal',
                            'Victor', 'Fernando', 'Horton', 'Tsung-han', 'Oshi', 'Alex', 'Kevin',
                            'Abram', 'shawn', 'TA', 'Ch', 'PB', 'Scotty', 'Bingbing'])
    female_names = frozenset(['Ashley', 'NC', 'Ella', 'Yu-tse', 'Mizu', 'MZ', 'GG', 'A'])
    if len(tokens) == 0:
        pass
    elif tokens[0] in male_names:
        return 1
    elif tokens[0] in female_names:
        return 2


if __name__ == '__main__':
    main()
