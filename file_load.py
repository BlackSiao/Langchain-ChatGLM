import nltk

words = nltk.word_tokenize('Python is a widely used programming language.')
print(nltk.pos_tag(words))
