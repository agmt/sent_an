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
	revs['word'] = []
	tree = ET.parse(infile)
	root = tree.getroot()
	stemmer = Stemmer.Stemmer('russian')
	
	for child in root:
		cur = {}
		cat = {cat.get('name'):cat.get('sentiment') for cat in child.find('categories') }['Food']
		if(cat == 'absence'):
			continue
		if(not(cat in revs['cat'])):
			revs['cat'].append(cat)
		cur['cat'] = cat
		
		txt = child.find('text').text.strip().lower().split()
		prevWord = ""
		stemmedReview = ""
		for word in txt:
			word = stemmer.stemWord(word.strip(",.!?:$\"()")) # 1 преобразование охуительнее другого
			if prevWord == "не" or prevWord == "":
				stemmedReview += word
			else:
				stemmedReview += " " + word
			prevWord = word
		cur['stext'] = stemmedReview
		cur['ltext'] = LabeledSentence(stemmedReview, ['%s_%s'%(labelType,len(revs['rev']))])
		
		words = stemmedReview.split()
		cur['words'] = {}
		for word in words:
			if(not(word in revs['word'])):
				revs['word'].append(word)
			cur['words'][word] = words.count(word)
		
	#	print("Cur:", cur)
	#	print("Words:", revs['word'])
		revs['rev'].append(cur)
	print("Parsed", len(revs['rev']), "reviews")
	return revs

def make_matrix(revs):
	rows = []
	cols = []
	data = []
	for i,rev in enumerate(revs['rev']):
		words = rev['words']
		rows.extend([i] * len(words))
		cols.extend(revs['word'].index(word) for word,val in words.items() if(word in revs['word']))
		data.extend(val for word,val in words.items())
	shape=(len(revs['rev']), len(revs['word']))
	matrix = csr_matrix((data,(rows, cols)), shape=shape)
	return matrix 

train = read("SentiRuEval_rest_markup_train.xml", "TRAIN")
matrix = make_matrix(train)
cats = [train['cat'].index(rev['cat']) for rev in train['rev']]

cls = MultinomialNB(alpha=0.1)
print(matrix.get_shape(), len(cats))
cls.fit(matrix, cats)
ans = cls.predict(matrix)
#ans = [random.randrange(0, len(train['cat'])) for rev in train['rev']]
answ = [train['cat'][cat] for cat in ans] 

precision_scores = f1_score(cats, ans, average=None)
for elem in precision_scores:
    print("{:.2f}".format(100 * elem), end=" ")
print("")
print("Average precision score: {:.2f}".format(100 *
    np.mean(precision_scores)))
