import re
from tflearn.data_utils import VocabularyProcessor

def regex_ops(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def clean_text(textList):
    """
    Funtion for cleaning of text from the garbage values.
    It uses regex_ops() function for performing regex operations.

    Parameters
    ----------
    textList: ndarray, required
        A numpy ndarray containing text sentences

    Returns
    -------
    cleanedTextList: ndarray
        A numpy ndarray of cleaned text sentences using regex operations
    """

    cleanedTextList = [regex_ops(sent) for sent in textList]
    return cleanedTextList

def vocab_generator(textList, window):
    return textVocab
