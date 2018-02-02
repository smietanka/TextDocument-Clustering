from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem.porter import PorterStemmer

wordsToStem = ['multiply', 'connecting', 'connected', 'said', 'provision', 'traditional', 'plotted', 'caresses']

stemWords = [LancasterStemmer().stem(word) for word in wordsToStem]
print(stemWords)
stemWords = [SnowballStemmer('english').stem(word) for word in wordsToStem]
print(stemWords)
stemWords = [PorterStemmer().stem(word) for word in wordsToStem]
print(stemWords)