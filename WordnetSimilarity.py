 from nltk.corpus import wordnet as wn
from nltk import word_tokenize as wt
from nltk.corpus import stopwords

def scoreline(line1,line2,metric,ic=None):
    sw = stopwords.words('english') # import stopwords
    t1 = wt(line1) # tokenize line1
    t2 = wt(line2) # tokenize line2
    syns1 = reduce(lambda x,y:x+y,[wn.synsets(x) for x in t1 if x not in sw]) # get list of synsets for tokens of line1
    syns2 = reduce(lambda x,y:x+y,[wn.synsets(x) for x in t2 if x not in sw]) # get list of synsets for tokens of line2
    runningscore = 0.0
    runningcount = 0
    print "syns1: ", syns1
    print "syns2: ", syns2
    for syn1 in set(syns1): # get Wordnet similarity score for <metric> for each pair created from both synset lists
        for syn2 in set(syns2):
            if ic is not None:
                try:
                    mark = metric(syn1,syn2)
                except:
                    mark = 0.0
                runningscore += mark
            else:
                try:
                    mark = metric(syn1,syn2)
                except:
                    mark = 0.0
            runningcount += 1

    score = runningscore/runningcount # add up individual scores, divide by number of individual scores
    return score # return overall scores


