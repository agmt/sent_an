#!/usr/bin/env python3

import re
from collections import defaultdict

import random
import bs4
import numpy as np
from nltk.stem import lancaster, porter
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB


def read_xml(infile):
    '''насчет soup= см. документацию bs4'''
    soup = bs4.BeautifulSoup(infile, "xml")
    root_tag = soup.find("Texts") #корневой тэг файла
    texts, categories, fileids = [], [], []
    #в тэг document входит text?  спросить?
    document_tags = list(root_tag.find_all("document"))
    #if shuffle:
    #    random.shuffle(document_tags)
    for document_tag in document_tags:
        category, fileid = document_tag['category'], document_tag['fileid']
        text_tag = document_tag.find("text")
        text = text_tag.string.strip()
        texts.append(text)
        categories.append(category)
        fileids.append(int(fileid))
    return texts, categories, fileids

def clear_text(text):
    #удаляет первый абзац, если он содержит только информацию об отправителя
    #то есть writesm wrote или article
    lines = [line.strip() for line in text.split("\n")]
    has_found = False
    for i, line in enumerate(lines):
        if line == "":
            has_found = True
            break
        #находим непустую строку?
        if has_found and i > 0 and bool(re.search("(article|writes|wrote)$",line[i-1])):
            lines = line[i+1:]
    return lines

class TextClassifier:

    STEMMERS = {'lancaster': lancaster.LancasterStemmer(),
                'porter': porter.PorterStemmer()}

    def __init__(self, min_count=1, max_count=np.inf,
                 max_categories_count=np.inf, stemming=None):
        self.min_count = min_count
        self.max_count = max_count
        # максимальное число категорий для слова
        self.max_categories_count = max_categories_count
        self.stemming = stemming
        self.word_codes = dict()
        self.dictionary_size = 0
        self.categories_encoding = dict()
        self.categories = []
        self.initialize()

    def initialize(self):
        if self.stemming is None:
            self.stemmer = None
        elif self.stemming in self.STEMMERS:
            self.stemmer = self.STEMMERS[self.stemming]
        else:
            raise KeyError("Unknown stemming mode {}".format(self.stemming))

    def preprocess(self, texts, categories=None):
        #texts --- список списков строк
        """
        перекодирует тексты в векторы так же, как это делалось
        при определении языка, используя в качестве признаков
        отдельные слова в тексте
        """
        # будем читать за один проход
        # каждый текст кодировать словарём {слово: число вхождений слова в текст}
        word_counts = defaultdict(int)
        word_counts_in_texts = defaultdict(int)
        words_in_texts = []
        # словарь для категорий, в которые входили слова
        if categories is not None:
            words_categories = defaultdict(set)
        for i, text in enumerate(texts):
            current_text_words = defaultdict(int)
            for line in text:
                splitted_line = line.strip().lower().split()
                for word in splitted_line:
                    word = word.strip(",.!?:$\"()")
                    if not word.isalpha() or len(word) == 0:
                        continue
                    if self.stemmer is not None:
                        word = self.stemmer.stem(word)
                    word_counts[word] += 1
                    current_text_words[word] += 1
            for word in current_text_words:
                word_counts_in_texts[word] += 1
            words_in_texts.append(current_text_words)
            if categories is not None:
                for word in current_text_words:
                    words_categories[word].add(categories[i])
        if categories is not None:
            # sorted_words_by_counts = sorted(word_counts_in_texts.items(),
            #                                 key=(lambda x: x[1]), reverse=True)
            # for word, count in sorted_words_by_counts[:100]:
            #     print(word, count)
            # sys.exit()
            # считаем для слова, в сколько категорий оно входило
            self.categories_counts_by_words =\
                {word: len(elem) for word, elem in words_categories.items()}
            # словарь, считающий, сколько слов встретилось в k категориях
            # counts_of_categories = defaultdict(int)
            # for word, count in self.categories_counts_by_words.items():
            #     counts_of_categories[count] += 1
            # print(" ".join(
            #     "{}:{}".format(key, count) for key, count
            #     in sorted(counts_of_categories.items())))
            # sys.exit()
            # self.word_counts_in_texts = word_counts_in_texts
            words = [word for word, count in word_counts.items()
                     if count >= self.min_count
                     and word_counts_in_texts[word] <= self.max_count
                     and self.categories_counts_by_words[word] <= self.max_categories_count]
            self.word_codes = {word: code for code, word in enumerate(words)}
            self.dictionary_size = len(self.word_codes)
        data, rows, cols = [], [], []
        for i, words_for_text in enumerate(words_in_texts):
            # words_for_text = {w_1: n_1, w_2: n_2, ...} -- словарь вхождений слов в текст
            current_vector_counts = defaultdict(int)
            for word, count in words_for_text.items():
                word_code = self.word_codes.get(word)
                if word_code is not None:
                    #current_vector_counts[word_code] =\
                    #    np.log2(1.0 + count / np.log2(1.0 + self.categories_counts_by_words[word]))
                    current_vector_counts[word_code] = count
            sorted_current_vector_counts = sorted(current_vector_counts.items())
            data.extend(x[1] for x in sorted_current_vector_counts)
            cols.extend(x[0] for x in sorted_current_vector_counts)
            rows.extend([i] * len(sorted_current_vector_counts))
        # if categories is not None:
        #     word_counts = defaultdict(int)
        #     for text in texts:
        #         for line in text:
        #             splitted_line = line.strip().split()
        #             for word in splitted_line:
        #                 word = word.strip(",.!?:$\"()")
        #                 if not word.isalpha() or len(word) == 0:
        #                     continue
        #                 if self.stemmer is not None:
        #                     word = self.stemmer.stem(word)
        #                 word_counts[word] += 1
        #                 # if word not in self.word_codes:
        #                 #     self.word_codes[word] = self.dictionary_size
        #                 #     self.dictionary_size += 1
        #     words = [word for word, count in word_counts.items()
        #              if count >= self.min_count]
        #     self.word_codes = {word: code for code, word in enumerate(words)}
        #     self.dictionary_size = len(self.word_codes)
        # data, rows, cols = [], [], []
        # for i, text in enumerate(texts):
        #     current_vector_counts = defaultdict(int)
        #     for line in text:
        #         splitted_line = line.strip().split()
        #         for word in splitted_line:
        #             word = word.strip(",.!?:$\"()")
        #             if not word.isalpha() or len(word) == 0:
        #                 continue
        #             if self.stemmer is not None:
        #                 word = self.stemmer.stem(word)
        #             word_code = self.word_codes.get(word)
        #             if word_code is not None:
        #                 current_vector_counts[word_code] += 1
        #     sorted_current_vector_counts = sorted(current_vector_counts.items())
        #     data.extend(x[1] for x in sorted_current_vector_counts)
        #     cols.extend(x[0] for x in sorted_current_vector_counts)
        #     rows.extend([i] * len(sorted_current_vector_counts))

        answer = csr_matrix((data,(rows, cols)), shape=(len(texts), self.dictionary_size))
        # print(len(data), len(cols), len(rows))
        recoded_categories = []
        if categories is not None:
            for category in categories:
                if category not in self.categories_encoding:
                    self.categories_encoding[category] = len(self.categories_encoding)
                    self.categories.append(category)
                recoded_categories.append(self.categories_encoding[category])
        if categories is not None:
            return answer, recoded_categories
        else:
            return answer


    def fit(self, texts, text_categories):
        X, y = self.preprocess(texts, text_categories)
        print(X.shape)
        self.cls = MultinomialNB(alpha=0.1)
        # self.cls = LogisticRegression()
        self.cls.fit(X, y)
        return self

    def fakepredict(self, texts):
        categories = [random.randrange(0, len(self.categories)) for text in texts]
        answer = [self.categories[elem] for elem in categories]
        return answer

    def predict(self, texts):
        X = self.preprocess(texts)
        categories = self.cls.predict(X)
        answer = [self.categories[elem] for elem in categories]
        return answer

def make_contingency_table(pred_labels, test_labels):
    counts = defaultdict(int)
    for elem in zip(test_labels, pred_labels):
        counts[elem] += 1
    return counts

if __name__ == "__main__":
    # args = sys.argv[1:]
    # if len(args) != 2:
    #     sys.exit
    # infile_train, infile_test = args
    infile_train = "train.xml"
    infile_test = "test.xml"
    with open(infile_train, "r", encoding="utf8") as fin:
        contents_train = fin.read()
    with open(infile_test, 'r', encoding='utf8') as fin:
        contents_test = fin.read()
    texts, categories, fileids = read_xml(contents_train)
    test_texts, test_categories, test_fileids = read_xml(contents_test)
    cls = TextClassifier(min_count=1, max_count=1000)
    clear_train = [clear_text(text) for text in texts]
    cls.fit(clear_train, categories)
    clear_test = [clear_text(test_text) for test_text in test_texts]
    answers = cls.predict(clear_test)
    print("Answers: ", answers)
    contingency_table = make_contingency_table(answers, test_categories)
    # for key, value in sorted(contingency_table.items()):
    #     print("{} {} {}".format(key[0], key[1], value))
    precision_scores = f1_score(test_categories, answers, average=None)
    for elem in precision_scores:
        print("{:.2f}".format(100 * elem), end=" ")
    print("")
    print("Average precision score: {:.2f}".format(100 *
        np.mean(precision_scores)))

