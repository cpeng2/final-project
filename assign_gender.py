
import collections
import nltk
assert nltk.download('punkt')
import jieba

# Make a list of the target text files
file1 = '0430_fieldnotes.txt'
file2 = '0408_fieldnotes.txt'
file3 = '0412_fieldnotes.txt'
file4 = '0430_fieldnotes.txt'
file5 = '0607_fieldnotes.txt'
files = [file1, file2, file3, file4, file5]

# Contains the chars we would like to ignore while processing the words
PUNCTUATION = '.,;!?#&-\'_+=/\\"@$^%()[]{}~: '


def main():
    # Read all files and add them to all_tokens
    all_tokens = []
    for file in files:
        with open(file, 'r') as source:
            for line in source:
                tokens = nltk.word_tokenize(line)
                # Parse Chinese words
                for token in tokens:
                    parsed_tokens = jieba.lcut(token)
                    all_tokens += parsed_tokens
    # Assign gender to words
    male_words = []
    female_words = []
    for token in all_tokens:
        if assign_gender(token) == 1:
            male_words.append(token)
        elif assign_gender(token) == 2:
            female_words.append(token)

    #count(male_words)
    count(male_words)


def count(all_tokens):
    frequencies = collections.Counter()
    frequencies.update(token.casefold() for token in all_tokens)
    for word, count in sorted(frequencies.items(), key=lambda t: t[1], reverse=False):
        if word == PUNCTUATION:
            pass
        print(f'{word}: {count}')


def assign_gender(tokens):

    male_names = ['N', 'CG', 'Nick', 'R', 'Rick', 'Mark', 'Joe', 'Me', 'Ronny', 'Gary', 'Bob', 'B', 'Brian']
    female_names = ['Ashley', 'NC', 'Ella']
    if len(tokens) == 0:
        pass
    elif tokens[0] in male_names:
        return 1
    elif tokens[0] in female_names:
        return 2


if __name__ == '__main__':
    main()
