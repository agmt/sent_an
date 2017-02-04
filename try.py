import numpy as np
import xml.etree.ElementTree as ET
import gensim
LabeledSentence = gensim.models.doc2vec.LabeledSentence
from sklearn.cross_validation import train_test_split
from gensim.models.word2vec import Word2Vec
import Stemmer

def read(infile, outall, outpos, outneg):
	tree = ET.parse(infile)
	root = tree.getroot()
	out = open(outall,"w")
	pos = open(outpos,"w")
	neg = open(outneg,"w")
	review_count = 0
	pos_review_count = 0
	neg_review_count = 0
	stemmer = Stemmer.Stemmer('russian')
	
	for child in root:
		#print(child.tag, child.attrib)
		stemmedReview = ""
		for i in child:
			if i.tag=="scores":
				for j in i:
					if j.tag=="food":
						foodscore = j.text
						if int(foodscore)>=6:
							foodscore = "positive"
						else:
							foodscore = "negative"
			if i.tag=="text":
				review = i.text.strip().lower().split()
				prevWord = ""
				for word in review:
					word = stemmer.stemWord(word.strip(",.!?:$\"()"))
					if prevWord == "не" or prevWord == "":
						stemmedReview += word
					else:
						stemmedReview += " " + word
					prevWord = word
		out.write(foodscore+"\n")
		if foodscore=="positive":
			pos.write(stemmedReview)
			out.write(stemmedReview)
			pos.write("\n")
			pos_review_count += 1
		else:
			neg.write(stemmedReview)
			out.write(stemmedReview)
			neg.write("\n")
			neg_review_count += 1
		out.write("\n")
		review_count += 1
	pos.close()
	neg.close()
	print(review_count)
	print(pos_review_count)
	print(neg_review_count)

read('SentiRuEval_rest_markup_train.xml', "reviews.txt", "pos.txt", "neg.txt")
read('SentiRuEval_rest_markup_test.xml', "reviews_test.txt", "pos_test.txt", "neg_test.txt")
		
with open('pos.txt','r') as infile:
    pos_reviews = infile.readlines()

with open('neg.txt','r') as infile:
    neg_reviews = infile.readlines()
	
y = np.concatenate((np.ones(len(pos_reviews)), np.zeros(len(neg_reviews))))
x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_reviews, neg_reviews)), y, test_size=0.2)

#Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
#We do this by using the LabeledSentence method. The format will be "TRAIN_i" or "TEST_i" where "i" is
#a dummy index of the review.
def labelizeReviews(reviews, label_type):
    labelized = []
    for i,v in enumerate(reviews):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized

x_train = labelizeReviews(x_train, 'TRAIN')
x_test = labelizeReviews(x_test, 'TEST')