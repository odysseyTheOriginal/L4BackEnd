from __future__ import division
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import brown
import math
import numpy as np
import sys

brown_freqs = dict()
N = 0

######################### word similarity ##########################

def get_best_synset_pair(word_1, word_2):
    """
    Choose the pair with highest path similarity among all pairs.
    Mimics pattern-seeking behavior of humans.

    """
    max_sim = -1.0
    synsets_1 = wn.synsets(word_1)
    synsets_2 = wn.synsets(word_2)
    if len(synsets_1) == 0 or len(synsets_2) == 0:
        return None, None
    else:
        max_sim = -1.0
        best_pair = None, None
        for synset_1 in synsets_1:
            for synset_2 in synsets_2:
                sim = wn.path_similarity(synset_1, synset_2)
                if sim == None:
                    sim = 0
                if sim > max_sim:
                    max_sim = sim
                    best_pair = synset_1, synset_2
        return best_pair


def length_dist(synset_1, synset_2):
    """
    sysnset represents a word
    Return a measure of the length of the shortest path in the semantic
    ontology between two synsets.

    """
    # integer larger than any practical  list or string index
    l_dist = sys.maxsize
    if synset_1 is None or synset_2 is None:
        return 0.0
    if synset_1 == synset_2:
        # if synset_1 and synset_2 are the same synset return 0
        l_dist = 0.0
    else:
        wset_1 = set([str(x.name()) for x in synset_1.lemmas()])
        wset_2 = set([str(x.name()) for x in synset_2.lemmas()])
        # assume they are similar from lemmas
        if len(wset_1.intersection(wset_2)) > 0:
            # if synset_1 != synset_2 but there is word overlap, return 1.0
            l_dist = 1.0
        else:
            # just compute the shortest path between the two
            l_dist = synset_1.shortest_path_distance(synset_2)
            if l_dist is None:
                l_dist = 0.0
    # normalize path length to the range [0,1]
    return math.exp(-0.2 * l_dist)


def hierarchy_dist(synset_1, synset_2):
    """
    Return a measure of depth in the ontology to model
    nodes closer to the root are broader and have less semantic similarity
    than nodes further away from the root.
    """
    h_dist = sys.maxsize
    if synset_1 is None or synset_2 is None:
        return h_dist
    if synset_1 == synset_2:
        # return the depth of one of synset_1 or synset_2
        h_dist = max([x[1] for x in synset_1.hypernym_distances()])
    else:
        # find the max depth of least common subsumer
        # ex: {Synset('object.n.01'): 4},, {Synset('person.n.01'): 0}.. if synset_1 is person
        hypernyms_1 = {x[0]: x[1] for x in synset_1.hypernym_distances()}
        hypernyms_2 = {x[0]: x[1] for x in synset_2.hypernym_distances()}

        # look for similar words
        lcs_candidates = set(hypernyms_1.keys()).intersection(set(hypernyms_2.keys()))

        if len(lcs_candidates) > 0:
            lcs_dists = []
            for lcs_candidate in lcs_candidates:
                lcs_d1 = 0
                if lcs_candidate in hypernyms_1:
                    lcs_d1 = hypernyms_1[lcs_candidate]
                lcs_d2 = 0
                if lcs_candidate in hypernyms_2:
                    lcs_d2 = hypernyms_2[lcs_candidate]
                # compare the distances for the same word from 2 sets and get the maximum
                lcs_dists.append(max([lcs_d1, lcs_d2]))
            # from all get the max, greater the distance higher value
            h_dist = max(lcs_dists)
        else:
            h_dist = 0
    # normalize the distance and return
    return ((math.exp(0.45 * h_dist) - math.exp(-0.45 * h_dist)) /
            (math.exp(0.45 * h_dist) + math.exp(-0.45 * h_dist)))


def word_similarity(word_1, word_2):
    """
    :param word_1: word in sentence
    :param word_2: each word in union

    """
    synset_pair = get_best_synset_pair(word_1, word_2)
    return (length_dist(synset_pair[0], synset_pair[1]) *
            hierarchy_dist(synset_pair[0], synset_pair[1]))


######################### sentence similarity ##########################

def most_similar_word(word, word_set):

    """
    Find the word in the joint word set that is most similar to the word
    passed in.
    word : word in phrase
    word_set : union of words

    """
    max_sim = -1.0
    sim_word = ""
    for ref_word in word_set:
        sim = word_similarity(word, ref_word)
        if sim > max_sim:
            max_sim = sim
            sim_word = ref_word
    return sim_word, max_sim


def info_content(lookup_word):

    """

    Uses the Brown corpus available in NLTK to calculate a Laplace
    smoothed frequency distribution of words, then uses this information
    to compute the information content of the lookup_word.

    lookup_word = word that is passed from the sentence

    """
    global N
    # N is the number of words
    if N == 0:
        # extract brown corpus sentences
        for sent in brown.sents():
            for word in sent:
                word = word.lower()
                # create a dictionary key is the word and value is the count
                if word not in brown_freqs:
                    brown_freqs[word] = 0
                brown_freqs[word] = brown_freqs[word] + 1
                N = N + 1
    lookup_word = lookup_word.lower()
    # get the frequency for the word if in brown
    n = 0 if lookup_word not in brown_freqs else brown_freqs[lookup_word]

    # probability of word over total words
    return 1.0 - (math.log(n + 1) / math.log(N + 1))


def semantic_vector(words, joint_words, info_content_norm):
    """

    Computes the semantic vector of a sentence.
    words = words of the sentence
    The size of the semantic vector = size of the joint word set.
    1 = word in sentence in wordset
    or the similarity of the word to the most similar word in the joint word set if it doesn't.
    info_content_norm = to normalize using brownbag information content (probability of words)

    """
    # set of words in sentence
    sent_set = set(words)
    # create an array of zeros of the length of words union
    semvec = np.zeros(len(joint_words))
    i = 0
    for joint_word in joint_words:
        if joint_word in sent_set:
            # if word in union exists in the sentence, s(i) = 1 (unnormalized)
            semvec[i] = 1.0
            if info_content_norm:
                # if content is normalized info content to the power 2
                semvec[i] = semvec[i] * math.pow(info_content(joint_word), 2)
        else:
            # find the most similar word in the joint set and set the sim value
            sim_word, max_sim = most_similar_word(joint_word, sent_set)
            semvec[i] = 0.2 if max_sim > 0.2 else 0.0
            if info_content_norm:
                semvec[i] = semvec[i] * info_content(joint_word) * info_content(sim_word)
        i = i + 1

    return semvec


def semantic_similarity(sentence_1, sentence_2, info_content_norm):
    """
    Computes the semantic similarity between two sentences as the cosine
    similarity between the semantic vectors computed for each sentence.
    """
    words_1 = nltk.word_tokenize(sentence_1)
    words_2 = nltk.word_tokenize(sentence_2)
    joint_words = set(words_1).union(set(words_2))

    # computing 2 vectors
    vec_1 = semantic_vector(words_1, joint_words, info_content_norm)
    vec_2 = semantic_vector(words_2, joint_words, info_content_norm)

    # cosine similarity calculation
    return np.dot(vec_1, vec_2.T) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))


######################### overall similarity ##########################

def similarity(sentence_1, sentence_2, info_content_norm):
    """
    Calculate the semantic similarity between two sentences. The last
    parameter is True or False depending on whether information content
    normalization is desired or not.
    """

    return semantic_similarity(sentence_1, sentence_2, info_content_norm)
