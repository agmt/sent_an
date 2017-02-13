#!/usr/bin/env python3

import numpy as np
import xml.etree.ElementTree as ET
import gensim
from gensim.models.doc2vec import LabeledSentence
from sklearn.cross_validation import train_test_split
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
from sklearn.naive_bayes import MultinomialNB,BernoulliNB

class Opt:
    wordMin = 0
    wordMax = 190
    wordRevMin = 23
    wordRevMax = 10000
    wordDropMin = 0
    wordDropMax = 5
    lenMin = 0
    conWords = set(('ед', 'вкус', 'невкус', 'кухн', 'аппетит'))
    conWidth = 5
    childCount = 30
    childVar = 1600

def getFit(train, test, words):
    train['word'] = {}
    for i,word in enumerate(words):
        train['word'][word] = int(i)
    train['word'] = words
    matrix,cats = make_matrix(train, train)
    mtest,ctest = make_matrix(test, train)

    cls = MultinomialNB(alpha=1e-8)
    cls.fit(matrix, cats)
    ans = cls.predict(mtest)

    precision_scores = f1_score(ctest, ans, average=None)
    print(precision_scores)
    return np.mean(precision_scores)

def evolve(words):
    res = []
    res.append(words)
    size = len(words)
    for i in range(Opt.childCount):
        cur = np.copy(words)
        for j in range(Opt.childVar):
            ind = random.randrange(size)
            cur[ind] = 1-cur[ind]
        res.append(cur)
    return res

def optimize(train, test):
    allWords = {ind:word for word,ind in train['word'].items()}
    parent = np.ones(len(allWords), dtype=np.int8)
    for big in range(1000):
        print("Iter=",big,Opt.childVar)
        if(big%100 == 0):
            Opt.childVar = int(Opt.childVar/2)
        childs = evolve(parent)
        childVal = []
        for child in childs:
            n = 0
            curWords = {}
            for ind,word in allWords.items():
                if(child[ind] == 1):
                    curWords[word] = n
                    n += 1
            fit = getFit(train, test, curWords)
            print("fit=",fit)
            print("words=",len(curWords))
            childVal.append(fit)
        childn = childVal.index(max(childVal))
        parent = childs[childn]
    return parent

def read(infile, labelType):
    revs = {}
    revs['rev'] = []
    revs['cat'] = []
    revswords = set()
    revs['drop'] = defaultdict(int)
    revs['wordCount'] = defaultdict(int)
    revs['wordCountOfRev'] = defaultdict(int)
    revs['wordByCategories'] = defaultdict(set)
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
        #  cat = 'positiv'
        #elif(catNum < 6):
        #  cat = 'negativ'
        #else:
        #  continue
        #if(cat == 'absence'):
        #    continue
        #if(cat == 'both'):
        #	continue
        if(cat == 'neutral'):
            continue
        if(cat == 'absence'):
            pass
        else:
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
        for i,word in enumerate(words):
            if(len(word) < Opt.lenMin):
                continue
            if(cat == 'absence'):
                revs['drop'][word] += 1
            else:
                cur['words'][word] += 1
                revs['wordCount'][word] += 1
                revs['wordByCategories'][word].add(cur['cat'])
                revswords.add(word)
                #if(word in Opt.conWords):
                #    context = words[max(i-Opt.conWidth, 0):(i+Opt.conWidth)]
                #    revswords.update(context)
        
        for word,val in cur['words'].items():
            revs['wordCountOfRev'][word] += 1
        
        if(cat == 'absence'):
            continue
        
        revs['rev'].append(cur)
    revs['word'] = {}
    for i,word in enumerate(revswords):
        revs['word'][word] = int(i)
    print("Parsed", len(revs['rev']), "reviews")
    return revs

def drop_words(revs):
    #revs['drop']['вкус'] = 10
    revswords = set(word for word,id in revs['word'].items()
                    if (Opt.wordRevMin <= revs['wordCountOfRev'][word] < Opt.wordRevMax)
                    and (Opt.wordMin <= revs['wordCount'][word] < Opt.wordMax)
                    and (revs['drop'][word] <= Opt.wordDropMax))
    revswords = set(['от', 'а', 'тольк', 'вкус', 'выбра', 'отмет', 'особен', 'цел', 'столик', 'сказа', 'из', 'ещ', 'неплох', 'хочет', 'выбор', 'сво', 'ег', 'девушк', 'приятн', '-', 'котор', 'то', 'праздник', 'как', 'можн', 'прост', 'перв', 'хот', 'горяч', 'посл', 'есл', 'мне', 'всё', 'общ', 'кухн', 'компан', 'посет', 'отличн', 'огромн', 'дан', 'обязательн', 'у', 'же', 'уютн', 'вам', 'над', 'обслужива', 'сам', 'вкусн', 'он', 'порадова', 'вообщ', 'ед', 'принесл', 'довольн', 'долг', 'мен', 'атмосфер', 'быстр', 'ну', 'один', 'их', 'персона', 'оказа', 'обслуживан', 'нет', 'гост', 'отдельн', 'сраз', 'понрав', 'готов', 'по', 'уж', 'напитк', 'интерьер'])
    revs['word'] = {}
    for i,word in enumerate(revswords):
        revs['word'][word] = int(i)

def make_matrix(revs, revsid):
    rows = []
    cols = []
    data = []
    for i,rev in enumerate(revs['rev']):
        words = {word:val for word,val in rev['words'].items() if(word in revsid['word'])}
        #words = {word: np.log2(1.0 + val / np.log2(1.0 + len(revsid['wordByCategories'][word]))) for word,val in rev['words'].items() if(word in revsid['word'])}
        rows.extend([i] * len(words))
        cols.extend(revsid['word'][word] for word,val in words.items())
        data.extend(val for word,val in words.items())
    shape=(len(revs['rev']), len(revsid['word']))
    matrix = csr_matrix((data,(rows, cols)), shape=shape)
    cats = [revsid['cat'].index(rev['cat']) for rev in revs['rev']]
    return matrix,cats

if __name__ == "__main__":
    train = read("SentiRuEval_rest_markup_train.xml", "TRAIN")
    test = read("SentiRuEval_rest_markup_test.xml", "TEST")
    #train = read("train.xml", "TRAIN")
    drop_words(train)
    print(getFit(train,test,train['word']))
    #allWords = {ind:word for word,ind in train['word'].items()}
    #wordFlags = optimize(train, test)
    #words = [word for ind,word in allWords.items() if wordFlags[ind] == 1]
    #badwords = [word for ind,word in allWords.items() if wordFlags[ind] == 0]
    print("")
    #print("GOOD!")
    #print(words)
    #print("BAD!")
    #print(badwords)
