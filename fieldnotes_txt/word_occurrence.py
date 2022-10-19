
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
	word_d = {}  # dictionary: you -> 100, the -> 100, ...jerry ->1
	all_tokens = []

	# Read all files and add them to all_tokens
	for file in files:
		with open(file, 'r') as f:
			for line in f:
				tokens = line.split()
				for token in tokens:
					parsed_tokens = jieba.lcut(token)
					all_tokens += parsed_tokens

	# Count words in all_tokens and save in word_d
	for token in all_tokens:
		token = string_manipulation(token)
		if token in word_d and token not in PUNCTUATION:
			# word_d[token] += 1 if not first time
			word_d[token] = word_d[token] + 1
		else:
			# First time
			word_d[token] = 1

	print_out_d(word_d)


# Remove punctuation and convert everything into lowercase
def string_manipulation(s):
	ans = ''
	for ch in s:
		if ch.isalpha() or ch.isdigit():
			ans += ch.lower()
	return ans


def print_out_d(d):
	"""This function prints out all the info in d"""
	for key, value in sorted(d.items(), key=lambda t: t[1], reverse=False):
		# [('you', 100), ('he',10),...], sort using t[1] value
		print(key, ':', value)


if __name__ == '__main__':
	main()
