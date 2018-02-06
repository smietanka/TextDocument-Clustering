import numpy as np
import pandas as pd
import csv
import re
import nltk
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
import seaborn as sn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import itertools
import operator
from collections import defaultdict
import html


stemmer = LancasterStemmer()
stopwords = set(stopwords.words('english'))

original_documents = []
all_documents = []
classes = []
haveClasses = True
testExcel = 'test'
folderName = 'ag_news_csv'
path = "D:\STUDIA\\Praca Inżynierska\Klasteryzacja\Dane testowe\\" + folderName + "\\" + folderName + "\\"
num_clusters = 4

class RowFromFile(object):
    def __init__(self, cluster=None, articleName=None, articleContent=None):
        self.cluster= cluster
        self.articleName = articleName
        self.articleContent = articleContent

def main():
    global classes
    if haveClasses:
        with open(path + "classes.txt") as f:
            classes = f.readlines()
        classes = [x.strip() for x in classes]

    with open(path + testExcel + ".csv", encoding="utf8") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in spamreader:
            if len(row) == 3:
                original_documents.append(RowFromFile(int(row[0]), row[1], row[2]))
    
    all_documents = [x.articleName + ' ' + x.articleContent for x in original_documents]

    new_list = [list(g) for k, g in itertools.groupby(sorted(original_documents, key=operator.attrgetter('cluster')), operator.attrgetter('cluster'))]
    clusterWithCountOfDocuments = []
    a = 1
    for it in new_list:
        clusterWithCountOfDocuments.append((a, len(it)))
        a += 1

    print('Quantity of text documents per cluster')
    print(clusterWithCountOfDocuments)

    # count tfidf
    tfidf = TfidfVectorizer(tokenizer=PrepareDocument)
    tfidfMatrix = tfidf.fit_transform(all_documents)

    # KMean - jakies wyniki daje
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=num_clusters, max_iter=400)
    km.fit(tfidfMatrix)
    clusters = km.labels_.tolist()
    #region Show popularies words in cluster
    cluster_WithTexts = {}
    k = 0
    for i in clusters:
        if i in cluster_WithTexts:
            cluster_WithTexts[i].append(original_documents[k].articleName + ' ' + original_documents[k].articleContent)
        else:
            cluster_WithTexts[i] = [original_documents[k].articleName + ' ' + original_documents[k].articleContent]
        k += 1

    cluster_most_words = {}
    for cluster, data in cluster_WithTexts.items():
        data = [PrepareDocument(x) for x in data]
        current_tfidf = TfidfVectorizer()
        current_tfidf_matrix = current_tfidf.fit_transform([' '.join(x) for x in data])

        current_tf_idfs = dict(zip(current_tfidf.get_feature_names(), current_tfidf.idf_))

        current_tuples = current_tf_idfs.items()
        cluster_most_words[cluster] = sorted(current_tuples, key = lambda x: x[1], reverse=True)[:5]

    for cluster, words in cluster_most_words.items():
        print('Cluster {0} key words: {1}'.format(cluster, words))
        print()
    #endregion
    
    resultObj = { 'title': all_documents, 'cluster': clusters }
    frame = pd.DataFrame(resultObj, index = [clusters] , columns = ['title', 'cluster'])

    originalPredictedClass = []
    with open(path + testExcel + "_result.csv", "w", newline='', encoding='utf8') as csvfile:
        wr = csv.writer(csvfile, delimiter=",",quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in frame.values.tolist():
            originalPredictedClass.append(int(row[1])+1)
            table = [str(row[1]), str('%s' % row[0])]
            wr.writerow(table)

    testClass = [x.cluster for x in original_documents]

    clustersWithOriginalClusters = {}
    k = 0
    for i in clusters:
        if i in clustersWithOriginalClusters:
            clustersWithOriginalClusters[i].append(original_documents[k].cluster)
        else:
            clustersWithOriginalClusters[i] = [original_documents[k].cluster]
        k += 1
    hist = plt.figure(2)
    clusterMaxesHistogram = []
    for key in sorted(clustersWithOriginalClusters.keys()):
        tempMaxes = defaultdict(int)
        for i in clustersWithOriginalClusters[key]:
            tempMaxes[i] += 1
        result = max(tempMaxes.items(), key=lambda x: x[1])
        clusterMaxesHistogram.append((key + 1, int(result[0])))
        plt.subplot(220 + 1 + key)
        plt.xlim([1, num_clusters])
        plt.hist(clustersWithOriginalClusters[key])
        plt.ylabel('Ilość dokumentów')
        plt.xlabel('Predykowane klasy do klasy testowej {0}'.format(key+1))
        plt.title('Klasa {0}'.format(key+1))
    hist.show()

    mappedClusters = mapClusters(originalPredictedClass, clusterMaxesHistogram, False)
    mappedtestClusters = mapClusters(testClass, clusterMaxesHistogram, True)

    ShowConfusionMatrix(testClass, originalPredictedClass, num_clusters, 3)
    ShowConfusionMatrix(testClass, mappedClusters, num_clusters, 4)

    allCasesPredictedClass = mapClustersForAllCases(originalPredictedClass, num_clusters)
    lastFig = plt.figure(1)
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
    lastFig.show()
    plt.show()

    return bool(1)

def ShowConfusionMatrix(testClass, predClass, num_clusters, figNum):
    fig = plt.figure(figNum)
    cnf_matrix = confusion_matrix(testClass, predClass)
    print(cnf_matrix)
    df_cm = pd.DataFrame(cnf_matrix, range(num_clusters), range(num_clusters))
    sn.set(font_scale=1)
    sn.heatmap(df_cm, annot=True, fmt='g', cmap='Blues')
    plt.ylabel('Etykieta rzeczywista')
    plt.xlabel('Etykieta predykowana')
    plt.title('{0} %'.format(round(accuracy_score(testClass, predClass) * 100, 2)))
    if haveClasses:
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=0)
        plt.yticks(tick_marks, classes, rotation=0)
    fig.show()

def PrepareDocument(document):
    documentWithoutHTMLTags = RemoveHTMLTags(document)
    documentWithoutUrl = RemoveUrl(documentWithoutHTMLTags)
    document = CleanDocument(documentWithoutUrl)
    tokenizedDoc = TokenizeDocument(document)
    removedDoc = RemoveStopWords(tokenizedDoc)
    stemmedDoc = StemDocument(removedDoc)
    return stemmedDoc

def TokenizeDocument(document):
    tokens = [word for sent in nltk.sent_tokenize(document) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            if len(token) >= 4:
                filtered_tokens.append(token)
    return filtered_tokens

def StemDocument(tokenizedDoc):
    return [stemmer.stem(t) for t in tokenizedDoc if t]

def CleanDocument(document):
    document = document.replace("\\n", " ")
    document = re.sub("[^a-zA-Z ]", " " , document)
    return document.lower()

def RemoveStopWords(tokenizedDocument):
    removedStopWords = [word for word in tokenizedDocument if not word in stopwords]
    return removedStopWords

def RemoveHTMLTags(document):
    unescapedDocument = html.unescape(document)
    return re.sub('<[^<]+?>', '', unescapedDocument)

def RemoveUrl(document):
    return re.sub(r'(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w\.-]*)*\/?\S', '', document, flags=re.MULTILINE)

def mapClustersForAllCases(predClass, ilosc_klas):
    result = []
    result.append(predClass)

    for numerKlasy in range(1, ilosc_klas):
        tempResult = []
        for i in predClass:
            predRealClassAfter = i + numerKlasy
            if(predRealClassAfter > ilosc_klas):
                predRealClassAfter = predRealClassAfter - ilosc_klas
            tempResult.append(predRealClassAfter)
        result.append(tempResult)
    return result

def mapClusters(predClass, newMap, testClass):
    result = []
    for clas in predClass:
        for map in newMap:
            if testClass:
                if map[0] == clas:
                    result.append(map[1])
            else:
                if map[1] == clas:
                    result.append(map[0])
    return result
main()