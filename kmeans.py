# https://gist.github.com/anabranch/48c5c0124ba4e162b2e3
import numpy as np
import pandas as pd
import csv
import re
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import seaborn as sn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

stemmer = SnowballStemmer("english")
stopwords = set(stopwords.words('english'))

class RowFromFile(object):
    def __init__(self, cluster=None, articleName=None, articleContent=None):
        self.cluster= cluster
        self.articleName = articleName
        self.articleContent = articleContent

def main():
    original_documents = []
    all_documents = []
    haveClasses = True
    testExcel = 'test2'
    folderName = 'ag_news_csv'
    path = "D:\STUDIA\\Praca Inżynierska\Klasteryzacja\Dane testowe\\" + folderName + "\\" + folderName + "\\"
    num_clusters = 4
    if haveClasses:
        with open(path + "classes.txt") as f:
            classes = f.readlines()
        classes = [x.strip() for x in classes]

    # read documents from file
    with open(path + testExcel + ".csv", encoding="utf8") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in spamreader:
            if len(row) == 3:
                all_documents.append(row[2])
                original_documents.append(RowFromFile(int(row[0]), row[1], row[2]))
        
    #clean documents
    all_documents = [cleanDocument(x) for x in all_documents]
    #tokenize documents
    all_documents = [TokenizeDocument(x) for x in all_documents]
    #stop words
    all_documents = [RemoveStopWords(x) for x in all_documents]
    #stem documents
    all_documents = [StemDocument(x) for x in all_documents]

    # count tfidf
    tfidf = TfidfVectorizer(norm='l2',min_df=1, use_idf=True)
    tfidfMatrix = tfidf.fit_transform(all_documents)
    terms = tfidf.get_feature_names()

    # KMean
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=num_clusters)
    km.fit(tfidfMatrix)
    clusters = km.labels_.tolist()


    #region Test
    cluster_WithTexts = {}
    k = 0
    for i in clusters:
        if i in cluster_WithTexts:
            cluster_WithTexts[i].append(original_documents[k].cluster)
        else:
            cluster_WithTexts[i] = [original_documents[k].cluster]
        k += 1

    cluster_most_words = {}
    # for cluster, data in cluster_WithTexts.items():
    #     #clean documents
    #     data = [cleanDocument(x) for x in data]
    #     #tokenize documents
    #     data = [TokenizeDocument(x) for x in data]
    #     #stop words
    #     data = [RemoveStopWords(x) for x in data]
    #     #stem documents
    #     #data = [StemDocument(x) for x in data]
    #     current_tfidf = TfidfVectorizer()
    #     current_tfidf_matrix = current_tfidf.fit_transform(data)

    #     current_tf_idfs = dict(zip(current_tfidf.get_feature_names(), current_tfidf.idf_))

    #     current_tuples = current_tf_idfs.items()
    #     cluster_most_words[cluster] = sorted(current_tuples, key = lambda x: x[1], reverse=True)[:5]

    # for cluster, words in cluster_most_words.items():
    #     print('Cluster {0} key words: {1}'.format(cluster, words))
    #     print()
    #endregion

    resultObj = { 'title': all_documents, 'cluster': clusters }
    frame = pd.DataFrame(resultObj, index = [clusters] , columns = ['title', 'cluster'])

    originalPredictedClass = []
    with open(path + testExcel + "_result.csv", "w", newline='', encoding='utf8') as csvfile:
        wr = csv.writer(csvfile, delimiter=",",quotechar='|', quoting=csv.QUOTE_MINIMAL)
        #for i in range(num_clusters):
        for row in frame.values.tolist():
            #add to actual class
            originalPredictedClass.append(int(row[1])+1)
            table = [str(row[1]), str('%s' % row[0])]
            wr.writerow(table)

    #http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    testClass = [x.cluster for x in original_documents]

    #region map_predicted_classes
    plt.figure(1)
    allCasesPredictedClass = dupa(originalPredictedClass, num_clusters)
    for index, predictedClass in enumerate(allCasesPredictedClass):
        cnf_matrix = confusion_matrix(testClass, predictedClass)
        print(cnf_matrix)

        df_cm = pd.DataFrame(cnf_matrix, range(num_clusters), range(num_clusters))
        
        plt.subplot(220 + index + 1)
        sn.set(font_scale=1)
        sn.heatmap(df_cm, annot=True, fmt='g', cmap='Blues')
        plt.ylabel('Etykieta rzeczywista')
        plt.xlabel('Etykieta predykowana')
        plt.title('{0} %'.format(round(accuracy_score(testClass, predictedClass) * 100, 2)))
        if haveClasses:
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=0)
            plt.yticks(tick_marks, classes, rotation=0)
    #endregion

    plt.show()
    return bool(1)

def TokenizeDocument(document):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(document) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            if len(token) > 5:
                filtered_tokens.append(token)
    return filtered_tokens

def StemDocument(document):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(document) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        #if re.search('[a-zA-Z]', token):
        filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens if t]
    return " ".join(stems)

def cleanDocument(document):
    #replace new line char to space
    document = document.replace("\\n", " ")
    #remove non alphabetical
    document = re.sub("[^a-zA-Z ]", " " , document)
    return document.lower()

def RemoveStopWords(tokenizedDocument):
    #stopwords
    removedStopWords = [word for word in tokenizedDocument if not word in stopwords]
    return " ".join(removedStopWords)

def dupa(predClass, ilosc_klas):
    result = []
    #dodajemy oryginal:
    # 1=1; 2=2; 3=3; 4=4
    result.append(predClass)

    #lecimy po ilosci klas (w tym przypadku 4, a ze glowna juz dodalismy do odejmujemy jedna iteracje)
    for numerKlasy in range(1, ilosc_klas):
        tempResult = []
        #lecimy po kazdym przypadku z predClass (czyli ta oryginalna wyjsciowa z agorytmu klasteryzacji)
        for i in predClass:
            predRealClassAfter = i + numerKlasy
            if(predRealClassAfter > ilosc_klas):
                predRealClassAfter = predRealClassAfter - ilosc_klas
            tempResult.append(predRealClassAfter)
        result.append(tempResult)
    return result

main()