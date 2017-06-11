import difflib


def similar_phrases(allPhrases,listToNetwork):
    """
    :param allPhrases: phrase list
    :param listToNetwork: list that will be passed to the front end
    :return: final list to front end
    """

    currentIndex = 0;

    for item in allPhrases:
        for index in range(len(allPhrases)):
            if (difflib.SequenceMatcher(None, item, allPhrases[index]).ratio() > 0.7) and (index > currentIndex):
                listToNetwork.append(allPhrases[index] + '|||' + item)
        currentIndex += 1


def formatData(dicItems, key_phrases):

    """

    :param list: input data in the form of a list
    :return: the data that will be passed to the front end in list format

    """

    listToNetwork = list()
    allPhrases = list()


    for key, value in dicItems.items():
        for each in value:
            for phrase in each:
                listToNetwork.append(key + '|||' + phrase)
        listToNetwork.append("Moodle|||" + key);

    for ind in key_phrases:
        for all in ind:
            allPhrases.append(all);

    similar_phrases(allPhrases,listToNetwork);

    return listToNetwork;









