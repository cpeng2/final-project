
import jieba

# Make a list of the target text files
file1 = '0430_fieldnotes.txt'
file2 = '0408_fieldnotes.txt'
file3 = '0412_fieldnotes.txt'
file4 = '0430_fieldnotes.txt'
file5 = '0607_fieldnotes.txt'
files = [file1, file2, file3, file4, file5]

# Contains the chars we would like to ignore while processing the words
PUNCTUATION = '.,;!?#&-\'_+=/\\"@$^%()[]{}~ '


def main():

	# Read all files and add them to all_tokens
	male_words = []
	female_words = []
	for file in files:
		with open(file, 'r') as sink:
			for line in sink:
				tokens = line.split()
				# parse the Chinese sentences
				parsed_chinese = parse_chinese(tokens)
				print(parsed_chinese)

				if assign_gender(parsed_chinese) == 1:
					male_words += [parsed_chinese]
				elif assign_gender(parsed_chinese) == 2:
					female_words += [parsed_chinese]

	print(male_words)
	print(female_words)

	# frequency_count(all_tokens)
	frequency_count(male_words)
	frequency_count(female_words)


# Parse Chinese sentences
def parse_chinese(tokens) -> [str]:
	parsed_sentences = []
	for token in tokens:
		parsed_token = jieba.lcut(token)
		parsed_sentences += [parsed_token]
	return parsed_sentences


def assign_gender(tokens):

	male_names = ['N', 'CG', 'Nick', 'R', 'Rick', 'Mark', 'Joe', 'Me', 'Ronny', 'Gary', 'Bob', 'B', 'Brian']
	female_names = ['Ashley', 'NC', 'Ella']

	if len(tokens) == 0:
		pass
	elif tokens[0] in male_names:
		return 1
	elif tokens[0] in female_names:
		return 2


def frequency_count(all_tokens):

	# Count words in all_tokens and save in word_d
	word_d = {}  # dictionary: you -> 100, the -> 100, ...jerry ->1
	for token in all_tokens:
		token = string_manipulation(token)
		if token in word_d and token not in PUNCTUATION:
			# word_d[token] += 1 if not first time
			word_d[token] = word_d[token] + 1
		else:
			# First time
			word_d[token] = 1
	print_out_d(word_d)


def string_manipulation(s):
	# Remove punctuation and convert everything into lowercase
	ans = ''
	for ch in s:
		if ch.isalpha() or ch.isdigit():
			ans += ch.lower()
	return ans


def print_out_d(d):
	"""
	: param d: (dict) key of type str is a word
					value of type int is the word occurrence
	---------------------------------------------------------------
	This method prints out all the info in d
	"""
	for key, value in sorted(d.items(), key=lambda t: t[1], reverse=False):
		# [('you', 100), ('he',10),...]
		print(key, '->', value)


if __name__ == '__main__':
	main()