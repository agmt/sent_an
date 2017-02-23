#!/usr/bin/env python3

import numpy as np
from gensim.models.doc2vec import LabeledSentence
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import Stemmer
import re
from collections import defaultdict
import random
import bs4
from nltk.stem import lancaster, porter
from scipy.sparse import csr_matrix
from gensim.models.doc2vec import Doc2Vec
from gensim.models.word2vec import Word2Vec


class Opt:
    """
    All options in 1 place
    """

    # Min and max count of words to use them
    wordMin = 0
    wordMax = 190
    # Min and max count of reviews with word to use this word
    wordRevMin = 23
    wordRevMax = 10000
    # Min and max count of word appearance in "Abscence" reviews to drop it
    wordDropMin = 0
    wordDropMax = 5
    # Minimal word legnth
    lenMin = 3
    #conWords = set(('ед', 'вкус', 'невкус', 'кухн', 'аппетит'))
    #conWidth = 1000
    # Child count and child variety for genetic algorithm
    childCount = 30
    childVar = 200

"""
Read xml with reviews into "Revs" class
@param infile input file name
@param labelType label for marking review
"""
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

"""
Select special words for using at next stage
"""
def drop_words(revs):
    #revs['drop']['вкус'] = 10
    revswords = set(word for word,id in revs['word'].items()
                    if (Opt.wordRevMin <= revs['wordCountOfRev'][word] < Opt.wordRevMax)
                    and (Opt.wordMin <= revs['wordCount'][word] < Opt.wordMax)
                    and (revs['drop'][word] <= Opt.wordDropMax)
                    and (Opt.lenMin <= len(word)))
    #revswords = set(['от', 'а', 'тольк', 'вкус', 'выбра', 'отмет', 'особен', 'цел', 'столик', 'сказа', 'из', 'ещ', 'неплох', 'хочет', 'выбор', 'сво', 'ег', 'девушк', 'приятн', '-', 'котор', 'то', 'праздник', 'как', 'можн', 'прост', 'перв', 'хот', 'горяч', 'посл', 'есл', 'мне', 'всё', 'общ', 'кухн', 'компан', 'посет', 'отличн', 'огромн', 'дан', 'обязательн', 'у', 'же', 'уютн', 'вам', 'над', 'обслужива', 'сам', 'вкусн', 'он', 'порадова', 'вообщ', 'ед', 'принесл', 'довольн', 'долг', 'мен', 'атмосфер', 'быстр', 'ну', 'один', 'их', 'персона', 'оказа', 'обслуживан', 'нет', 'гост', 'отдельн', 'сраз', 'понрав', 'готов', 'по', 'уж', 'напитк', 'интерьер'])
    #revswords = set(['во-втор', 'проход', '19.04', 'незнаком', 'быстро...д', 'вследств', 'лов', 'тот', 'отмет', 'буженин', 'минут30', 'медлительн', 'никакая,какой-т', 'допинг', 'жител', 'застол', 'у', 'коктейл', 'нельз', 'сварен', 'опя', 'закуски!горяч', 'танцевальн', 'решен', 'перв', 'цел', 'кром', 'профессионализм', 'прежд', 'нужн', 'гроссенштрасс', 'приготовлен', 'неоправда', 'он', 'отличн', 'неожида', 'меню,м', 'конждиционер', 'проведён', 'скрыт', 'солянк', 'нет', 'зов', 'совс', 'выдел', 'созванив', 'перец,свеж', 'принесут', 'начита', 'общ', 'брон', 'сто', 'а', 'сво', 'производ', 'человека,ед', 'большие,н', 'оказа', 'вниман', 'обслуживан', '4-х', 'улучша', 'по', 'выполн', 'попрост', 'дива', 'скис', 'неплох', 'сраз', 'порадова', 'готов', 'разочаровали,посад', 'хот', 'но', 'неможет', '%', 'бесед', 'полон', 'копейк', 'нерозов', 'то', 'невероятн', 'кубик', 'выбра', 'вдол', 'гардеробщик', '16', 'несомнев', 'маловат', 'уютн', 'помо', 'всю', 'вол', 'котор', 'дан', 'жених', 'порядк', 'лазан', 'судак', 'непришл', 'нелучш', '9', 'столиков-', 'бульон', 'имх', 'понрав', 'есл', 'от', 'прост', 'ввел', 'детальн', 'хочет', 'мне', 'официант', 'ег', 'памя', 'натал', 'приятн', 'мякот', 'дальш', 'немен', 'хачапур', 'несл', 'расхлябан', 'долг', 'непомн', 'гост', 'непорт', 'потрат', '-', 'офис', 'вам', 'необходим', 'чизкейк', 'тормознут', 'попада', 'праздник', 'довольны,спасиб', 'зач', 'быстр', 'бахр', 'ну', 'детьм', 'грец', 'порасторопн', 'гармоничн', 'ненапряга', 'мен', 'неч', 'довольн', 'вкусный)', 'бока', 'манил', 'отдельн', 'дым', 'приватн', 'красот', 'посмотрет', 'довольна,продукт', 'ед', 'кафешк', 'выбор', 'приносили-унос', 'обслуг', 'водк', 'черн', 'напитк', 'продолжа', 'вкус', 'небуд', 'предварительн', 'предупреж', 'успех', 'кухн', 'обеден', '23.12.2013г.провел', 'признак', 'подушк', 'огромн', 'посл', 'вернут', 'нечт', 'благ', 'лиц', 'лук', 'принесл', 'рождения,вс', 'несдава', 'как', 'уютно,б', 'уж', 'суховат', 'эрмитаж', 'быстро...посл', '260р', 'столик', 'предлага', 'испарились,естествен', 'сцен', 'рискнуть,', 'же', 'музык', 'принима', 'из', '10,поболта', '550', 'сидел', 'всё', 'меньш', 'расслабля', 'персона', 'непоеха', 'атмосфер', 'четыр', 'уютно!н', 'пахнет', 'замороженные,вкус', 'тает', 'истор', 'т.р', 'неперв', 'ненаполня', 'оценива', '400', 'смягч', 'неусомн', 'прогуля', 'подолг', 'толик', 'ав', 'пакет', 'самоустран', 'недопустим', 'замучен', 'анастасия),приветлив', 'приятно,когд', 'оксане-', 'компан', 'пожела', 'тон', 'вкусн', 'сырн', 'пела),', 'один', 'повесел', 'понравилось,обслуживан', 'женщин', '8-го', 'говяж', 'растянул', 'день,веранд', 'замеча', 'сам', 'подвод', 'бело-розов', 'ноябр', 'однокурсник', 'предостаточн', 'несобра', 'девушк', 'спрашива', 'их', 'сказа', 'складыва', 'тёпло', 'рассмотрен', 'хорошос', 'блюдо(599', '5-7', 'неотказа', 'можн', 'интерьер', 'вообщ', 'украш', 'горяч', 'кушали(пробк', '15', 'ещ', 'тольк', 'послевкус', 'обслужива', 'дружеск', 'салат', 'обязательн', 'съесть,настольк', 'качествен', 'приготов', 'неподход', 'увидет', 'ресторан,', 'кругл', 'восем', 'познаком', 'присел', 'над', 'существен', 'посет', 'пада', 'хамств', 'но,т', 'назначен', 'свадебн'])

    revs['word'] = {}
    for i,word in enumerate(revswords):
        revs['word'][word] = int(i)

"""
Create matrix of features
@param revs reviews in "Revs"-structure
@param revsid structure with word ids
"""
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


"""
Get quality of prediction
@param train train reviews
@param test test reviews
@words used words in enumerable container
"""
def getFit(train, test, words):
    train['word'] = {}
    for i,word in enumerate(words):
        train['word'][word] = int(i)
    train['word'] = words
    matrix,cats = make_matrix(train, train)
    mtest,ctest = make_matrix(test, train)

    #matrix = preprocessing.normalize(matrix).toarray()
    #mtest = preprocessing.normalize(mtest).toarray()
    model = MultinomialNB(alpha=1e-8)
    #model = GaussianNB()
    #model = ExtraTreesClassifier()
    #model = LogisticRegression()
    #model = KNeighborsClassifier() # Very bad
    #model = DecisionTreeClassifier()
    #model = SVC()
    
    model.fit(matrix, cats)
    #print(model.feature_importances_)
    
    ans = model.predict(mtest)
    
    print(metrics.classification_report(ctest, ans))
    print(metrics.confusion_matrix(ctest, ans))
    
    precision_scores = metrics.f1_score(ctest, ans, average=None)
    print(precision_scores)
    return np.mean(precision_scores)

def genetic(train, test):
    Opt.lenMin = 0
    allWords = train['word']
    drop_words(train)
    bestWords = train['word']
    train['word'] = allWords
    parent = np.zeros(len(allWords), dtype=np.int8)
    for word,ind in bestWords.items():
        parent[allWords[word]] = 1
    train['word'] = allWords
    allWords = {ind:word for word,ind in train['word'].items()}
    wordFlags = optimize(train, test, parent)
    words = [word for ind,word in allWords.items() if wordFlags[ind] == 1]
    badwords = [word for ind,word in allWords.items() if wordFlags[ind] == 0]
    print("")
    print("GOOD!")
    print(words)
    print("BAD!")
    print(badwords)

def evolve(words):
    res = []
    res.append(np.copy(words))
    size = len(words)
    for i in range(Opt.childCount):
        cur = np.copy(words)
        for j in range(Opt.childVar):
            ind = random.randrange(size)
            cur[ind] = 1-cur[ind]
        res.append(cur)
    return res

def optimize(train, test, parent):
    allWords = {ind:word for word,ind in train['word'].items()}
    for big in range(1000):
        print("Iter=",big,Opt.childVar)
        if(big%100 == 0):
            Opt.childVar = int((Opt.childVar+1)/2)
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

def predict(train, test):
    drop_words(train)
    print(getFit(train,test,train['word']))

if __name__ == "__main__":
    #model = Word2Vec.load_word2vec_format('~/Downloads/word2vec/GoogleNews-vectors-negative300.bin', binary=True)
    #print(model['computer'])
    train = read("SentiRuEval_rest_markup_train.xml", "TRAIN")
    test = read("SentiRuEval_rest_markup_test.xml", "TEST")
    predict(train, test)
