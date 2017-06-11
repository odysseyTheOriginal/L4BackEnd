
# import nltk
# from nltk.corpus import wordnet
#
# lemma = nltk.wordnet.WordNetLemmatizer()
# # st = LancasterStemmer()
#
# wordlist =['data', 'model', 'modeling', 'models']
# newwordlist = [];
#
# for word in wordlist:
#      # newwordlist.append(st.stem(word))
#      newwordlist.append(lemma.lemmatize(word, wordnet.VERB))
#      # lemmatizer.lemmatize('going', wordnet.VERB)
# print(set(newwordlist));




# code to extract top 5 unique phrases

import difflib

# wordlist =['model', 'modeling', 'models', 'data models','database', 'dbms','udari', 'jalli', 'management', 'done', 'project']
# uniqueWordList = [];
# duplicates = [];

# def createuniquelist(wordlist):
#     currentindex = 0;
#     count = 0;
#     for word in wordlist:
#         for index in range(len(wordlist)):
#             if (difflib.SequenceMatcher(None, word, wordlist[index]).ratio() > 0.7) and (index > currentindex):
#                 duplicates.append(wordlist[index]);
#         if(word not in duplicates):
#             if count < 5:
#                 uniqueWordList.append(word);
#             count = count +1 ;
#         currentindex += 1;
#     print(uniqueWordList);
#
# createuniquelist(wordlist);




# create a dictionary of lists

from collections import defaultdict

# wordlist =[['a','b','c'],['aaa','bbb','ccc']]
# doclist= ['pdf1','pdf2']
#
# d = defaultdict(list)
# for k,v in zip(doclist,wordlist):
#     d[k].append(v)
#
# # print(d)
# print(d.items());
# print(d.get("pdf2"))

# d = defaultdict(list)
# wordlist =[['a','b','c'],['aaa','bbb','ccc']]
# doclist= ['pdf1','pdf2']
#
# for i,j in zip(wordlist,doclist):
#         d[j].append(i);
#
# print(d.get(doclist[1]))



# stemming in python
from nltk import PorterStemmer

text = ["A compiler translates code from a source language","database modelling" , "no way this is done"]
# for word in text.split(" "):
#     PorterStemmer().stem_word(word)

# print(PorterStemmer().stem('complications'))
newText = list();
for phrase in text:
        newText.append(" ".join(PorterStemmer().stem(word) for word in phrase.split(" ")))
print(newText)









