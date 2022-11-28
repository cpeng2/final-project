
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
    # Read all files and add them to all_tokens
    all_tokens = []
    for file in files:
        with open(file, 'r') as source:
            for line in source:
                if re.search(pattern, line, re.IGNORECASE):
                    tokens = nltk.word_tokenize(line)
                    # Parse Chinese words
                    for token in tokens:
                        parsed_tokens = jieba.lcut(token)
                        all_tokens += parsed_tokens
    # Count words
    punctuation = frozenset(string.punctuation)
    frequencies = collections.Counter()
    frequencies.update(token.casefold() for token in all_tokens if token not in punctuation)
    for word, count in sorted(frequencies.items(), key=lambda t: t[1], reverse=False):
        print(f'{word}: {count:,}')


if __name__ == '__main__':
    main()
