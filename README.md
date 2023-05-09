# final-project

This project counts word occurrences in my field notes. The field notes were taken at NYC Mandarin/ English meetups, where Mandarin learners come to practice with native speakers. The goals of this project are 1) count overall word occurrences, 2) count word occurrences used by males and females, and 3) label positive and negative words to conduct a sentiment analysis (stretch goal).

Updates:

5/9/2023
1) Used key words, code-switching, target language use, and polarity scores as features to predict the gender of a speaker.
2) Used `DictVectorizor`, `CountVectorizor` and `TF-IDF` to extract features and compare the restuls with different models.

12/15/2022
1) Used `match.group` to extract gender and direct quote
2) Removed punctuation before using `.extend` to add tokens to the list
3) Replaced `.sort` with `.most_common()`

12/05/2022
1) Names are now set to be `frozenset.`
2) Punctuations are removed using `string.punctuation`.
3) Regular expressions were added to extract the actual quotes.
4) The codes for `assign_gender` were revised, where I also left a question.
5) I'm still working on how to use `vaderSentiment`.
