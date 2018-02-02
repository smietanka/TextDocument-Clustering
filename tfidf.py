import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
def display_scores(vectorizer, tfidf_result):
    scores = zip(vectorizer.get_feature_names(), np.asarray(tfidf_result.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    allTerms = ["{0} ({1})".format(z[0], z[1]) for z in sorted_scores]
    for term in allTerms:
        print(term)
text1 = "The game of life is a game of everlasting learning"
text2 = "The unexamined life is not worth living"
text3 = "Never stop learning"
tfidf = TfidfVectorizer()
tfidfMatrix = tfidf.fit_transform([text1, text2, text3])
display_scores(tfidf, tfidfMatrix)