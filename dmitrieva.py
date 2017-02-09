#!/usr/bin/env python3

import numpy as np
import xml.etree.ElementTree as ET
import gensim
from gensim.models.doc2vec import LabeledSentence
from sklearn.model_selection import train_test_split
from gensim.models.doc2vec import Doc2Vec
from gensim.models.word2vec import Word2Vec
import Stemmer
import re
from collections import defaultdict
import random
import bs4
from nltk.stem import lancaster, porter
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB

def read(infile, labelType):
    revs = {}
    revs['rev'] = []
    revs['cat'] = []
    revswords = set()
    revs['wordCount'] = defaultdict(int)
    revs['wordCountOfRev'] = defaultdict(int)
    stemmer = Stemmer.Stemmer('russian')

    data = open(infile, "r", encoding="utf8").read()
    tree = bs4.BeautifulSoup(data, "xml")
    root = tree.find("reviews").find_all('review')
    #root = tree.find("Texts").find_all('document')
    for child in root:
        cur = {}

        cur['id'] = int(child['id'])
        cat = {cat['name']:cat['sentiment'] for cat in child.find('categories').find_all('category') }['Food']
        #cur['id'] = int(child['fileid'])
        #cat = child['category']

        #catNum = int(child.find('scores').find('food').text)
		#if(catNum > 8):
		#	cat = 'positiv'
		#elif(catNum < 6):
		#	cat = 'negativ'
		#else:
		#	continue
        if(cat == 'absence'):
            continue
        #if(cat == 'both'):
        #	continue
        if(cat == 'neutral'):
            continue
        if(not(cat in revs['cat'])):
            revs['cat'].append(cat)
        cur['cat'] = cat
		
        txt = child.find('text').text.lower().split()
        prevWord = ""
        stemmedReview = ""
        for word in txt:
            word = stemmer.stemWord(word.strip(",.!?:$\"()"))
            #word = word.strip(",.!?:$\"()")
            if((prevWord == 'не') or ( prevWord== "")):
                stemmedReview += word
            else:
                stemmedReview += " " + word
            prevWord = word
        cur['stext'] = stemmedReview
        cur['ltext'] = LabeledSentence(stemmedReview, ['%s_%s'%(labelType,len(revs['rev']))])
		
        words = stemmedReview.split()
        cur['words'] = defaultdict(int)
        cur['wordByCategories'] = {}
        for word in words:
            if(len(word) < 3):
                continue
            revswords.add(word)
            cur['words'][word] += 1
            revs['wordCount'][word] += 1
        for word,val in cur['words'].items():
            revs['wordCountOfRev'][word] += 1

        revs['rev'].append(cur)
    revs['word'] = {}
    for i,word in enumerate(revswords):
        revs['word'][word] = int(i)
    print("Parsed", len(revs['rev']), "reviews")
    return revs

def drop_words(revs):
    revswords = set(word for word,id in revs['word'].items() if revs['wordCountOfRev'][word] > 25 and revs['wordCount'][word] < 100)
    revs['word'] = {}
    for i,word in enumerate(revswords):
        revs['word'][word] = int(i)

def make_matrix(revs, revsid):
    rows = []
    cols = []
    data = []
    for i,rev in enumerate(revs['rev']):
        words = {word:val for word,val in rev['words'].items() if(word in revsid['word'])}
        #current_vector_counts[word_code] = np.log2(1.0 + count / np.log2(1.0 + self.categories_counts_by_words[word]))
        rows.extend([i] * len(words))
        cols.extend(revsid['word'][word] for word,val in words.items())
        data.extend(val for word,val in words.items())
    shape=(len(revs['rev']), len(revsid['word']))
    matrix = csr_matrix((data,(rows, cols)), shape=shape)
    cats = [revsid['cat'].index(rev['cat']) for rev in revs['rev']]
    return matrix,cats

if __name__ == "__main__":
    train = read("SentiRuEval_rest_markup_train.xml", "TRAIN")
    #train = read("train.xml", "TRAIN")
    drop_words(train)
    matrix,cats = make_matrix(train, train)

    cls = MultinomialNB(alpha=0.1)
    print(matrix.get_shape(), len(cats))
    cls.fit(matrix, cats)

    test = read("SentiRuEval_rest_markup_test.xml", "TEST")
    #test = read("test.xml", "TEST")
    mtest,ctest = make_matrix(test, train)
    print(mtest.get_shape(), len(ctest))
    ans = [var for var in cls.predict(mtest)]
    #ans = [random.randrange(0, len(train['cat'])) for rev in train['rev']]
    #print([train['cat'][cat] for cat in ctest])
    #print([train['cat'][cat] for cat in ans])
    #print([cat for cat in ctest])
    #print([cat for cat in ans])

    precision_scores = f1_score(ctest, ans, average=None)
    for elem in precision_scores:
        print("{:.2f}".format(100 * elem), end=" ")
    print("")
    print("Average precision score: {:.2f}".format(100 *
        np.mean(precision_scores)))
