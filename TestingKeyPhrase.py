import re
import string
import operator
import numpy as np
from unidecode import unidecode
from nltk import word_tokenize, sent_tokenize
from nltk import pos_tag_sents
from nltk.chunk.regexp import RegexpParser
from nltk.chunk import tree2conlltags
from nltk.corpus import stopwords
from itertools import chain, groupby
from operator import itemgetter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet as wn
from nltk.stem.lancaster import LancasterStemmer
import difflib
from collections import defaultdict
import json
import re

punct_re = re.compile('[{}]'.format(re.escape(string.punctuation)))
stop_words = set(stopwords.words('english'))

def generate_candidate(texts, method='phrase', remove_punctuation=True):
    """
    Generate word candidate from given string

    Parameters
    ----------
    texts: str, input text string
    method: str, method to extract candidate words, either 'word' or 'phrase'

    Returns
    -------
    candidates: list, list of candidate words
    """
    words_ = list()
    candidates = list()

    # tokenize texts to list of sentences of words
    sentences = sent_tokenize(texts)
    for sentence in sentences:
        if remove_punctuation:
            sentence = punct_re.sub(' ', sentence) # remove punctuation
            # sentence = re.sub(r'[^\w]', ' ', sentence)
        words = word_tokenize(sentence)
        words = list(map(lambda s: s.lower(), words))
        words_.append(words)
    tagged_words = pos_tag_sents(words_) # POS tagging

    if method == 'word':
        tags = set(['JJ', 'JJR', 'JJS', 'NN', 'NNP', 'NNS', 'NNPS'])
        tagged_words = chain.from_iterable(tagged_words)
        for word, tag in tagged_words:
            if tag in tags and word.lower() not in stop_words:
                candidates.append(word)
    elif method == 'phrase':
        grammar = r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'
        chunker = RegexpParser(grammar)
        all_tag = chain.from_iterable([tree2conlltags(chunker.parse(tag)) for tag in tagged_words])
        for key, group in groupby(all_tag, lambda tag: tag[2] != 'O'):
            candidate = ' '.join([word for (word, pos, chunk) in group])
            if key is True and candidate not in stop_words:
                candidates.append(candidate)
    else:
        print("Use either 'word' or 'phrase' in method")
    return candidates


def keyphrase_extraction_tfidf(texts, method='phrase', min_df=5, max_df=0.8, num_key=15):
    """
    Use tf-idf weighting to score key phrases in list of given texts

    Parameters
    ----------
    texts: list, list of texts (remove None and empty string)

    Returns
    -------
    key_phrases: list, list of top key phrases that expain the article

    """
    print('generating vocabulary candidate...')
    vocabulary = [generate_candidate(unidecode(text), method=method) for text in texts]
    vocabulary = list(chain(*vocabulary))
    vocabulary = list(np.unique(vocabulary)) # unique vocab
    print('done!')

    max_vocab_len = max(map(lambda s: len(s.split(' ')), vocabulary))
    tfidf_model = TfidfVectorizer(vocabulary=vocabulary, lowercase=True,
                                  ngram_range=(2,max_vocab_len), stop_words=None,
                                  min_df=min_df, max_df=max_df)
    X = tfidf_model.fit_transform(texts)
    vocabulary_sort = [v[0] for v in sorted(tfidf_model.vocabulary_.items(),
                                            key=operator.itemgetter(1))]
    sorted_array = np.fliplr(np.argsort(X.toarray()))

    # return list of top candidate phrase
    key_phrases = list()
    for sorted_array_doc in sorted_array:
        key_phrase = [vocabulary_sort[e] for e in sorted_array_doc[0:num_key]]
        key_phrases.append(key_phrase)

    return key_phrases


# def freqeunt_terms_extraction(texts, ngram_range=(1,1), n_terms=None):
#     """
#     Extract frequent terms using simple TFIDF ranking in given list of texts
#     """
#     tfidf_model = TfidfVectorizer(lowercase=True,
#                                   ngram_range=ngram_range, stop_words=None,
#                                   min_df=5, max_df=0.8)
#     X = tfidf_model.fit_transform(texts)
#     vocabulary_sort = [v[0] for v in sorted(tfidf_model.vocabulary_.items(),
#                                             key=operator.itemgetter(1))]
#     ranks = np.array(np.argsort(X.sum(axis=0))).ravel()
#     frequent_terms = [vocabulary_sort[r] for r in ranks]
#     frequent_terms = [f for f in frequent_terms if len(f) > 3]
#     return frequent_terms_filter

uniqueWordList = [];
duplicates = [];

def createuniquelist(wordlist):

    """
    :param wordlist: insert a list of strings to be filtered
    :return: returns a list of max 5 elements (after filtering similar words)

    """
    uniqueWordList=[]
    currentindex = 0;
    count = 0;
    for word in wordlist:
        for index in range(len(wordlist)):
            if (difflib.SequenceMatcher(None, word, wordlist[index]).ratio() > 0.7) and (index > currentindex):
                duplicates.append(wordlist[index]);
        if(word not in duplicates):
            if count < 5:
                uniqueWordList.append(word);
            count += 1
        currentindex += 1
    # print(uniqueWordList);
    return uniqueWordList


if __name__ == '__main__':
    import pandas as pd
    # texts = list(pd.read_csv('data/example.txt')['abstract'])

    rankedDocuments = ['pdf/IntroductionSCM.txt'];
    texts = list();
    for doc in rankedDocuments:
        with open(doc, encoding="utf8") as f:
            sample = f.read()
            # sample = re.sub('[!@#$*-•]', '', sample)
            # print(sample)
            texts.append(sample);

    key_phrases = keyphrase_extraction_tfidf(texts)

    # # print(key_phrases);
    # print(*key_phrases, sep='\n')
    # # print(wn.synsets("hot"));

    # to get the maximum of 5 key phrases and put it to a dictionary where the key is the document name and the values are a list of keyphrases
    d = defaultdict(list)
    returnfromuniqueqordlist=[]
    for ind in key_phrases:
        returnfromuniqueqordlist.append(createuniquelist(ind))

    for i, j in zip(returnfromuniqueqordlist, rankedDocuments):
        d[j].append(i);

    # pass the dictionary as a list to frontend
    passinglist = list(d.items())
    print(passinglist)

    #convert python to json data
    # jsonobj= json.dumps(d);
    # print(jsonobj)












