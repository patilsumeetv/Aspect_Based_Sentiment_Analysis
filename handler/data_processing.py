import re
import numpy as np
import tensorflow as tf

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
    # max sentence length
    mlen = 100
    # vocab processor object
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(mlen)
    # converting words to indices
    textVocab = list(vocab_processor.fit_transform(textList))
    # setting vocab size
    vocab_size = len(vocab_processor.vocabulary_)
    # converting numpy to python list
    textVocab = [i.tolist() for i in textVocab]

    # getting rid of trailing 0's
    for lst in textVocab:
        try:
            while lst[-1] == 0:
                lst.pop()
        except:
            pass

    # filter the empty list
    textVocab = filter(None, textVocab)
    # converting python to numpy list
    textVocab = np.array(list(textVocab))

    # pivot words i.e. our inputs
    pivot_words = []
    # target words i.e. our outputs
    target_words = []
    # iterating though each sentences
    for d in range(textVocab.shape[0]):
        pivot_idx = textVocab[d][window:-window]

        for i in range(len(pivot_idx)):
            # getting current pivot word
            pivot = pivot_idx[i]
            # target array
            targets = np.array([])
            # negative & positive targets
            neg_target = textVocab[d][i:window+i]
            pos_target = textVocab[d][i+window+1:i+window+window+1]
            targets = np.append(targets, [neg_target, pos_target]).flatten().tolist()

            for c in range(window*2):
                pivot_words.append(pivot)
                target_words.append(targets[c])

    return pivot_words, target_words, vocab_size
