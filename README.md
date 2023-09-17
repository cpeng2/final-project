# final-project

This project uses transcribed text data and machine learning models to predict the gender of a second language speaker. The text data are the field notes taken at NYC Mandarin/ English meetups, where Mandarin learners come to practice with native speakers. The goals of this project are 1) to explore gender differences in second language use, including word frequency, code-switching, ratio of target language use, and sentiment, and 2) to build a computational model using these hypotheses as features to predict the gender of the speaker.

Updates:

9/17/2023
1) None of the features seem to be significant predictors of gender.
2) I'd like to modify the current features and hopefully add more features to improve the model.
3) Fine-tune the parameters and evaluate the model.

5/9/2023
1) Used keywords, code-switching, target language use, and polarity scores as features to predict the gender of a speaker.
2) Used `DictVectorizor`, `CountVectorizor` and `TF-IDF` to extract features and compare the results with different models.

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
