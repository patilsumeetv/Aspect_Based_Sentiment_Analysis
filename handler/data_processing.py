import re

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

def clean_reviews(reviewList):
    """
    Funtion for cleaning of review text from the garbage values.
    It uses regex_ops() function for performing regex operations.

    Parameters
    ----------
    reviewList: ndarray, required
        A numpy ndarray containing review sentences

    Returns
    -------
    cleanedReviewList: ndarray
        A numpy ndarray of cleaned review sentences using regex operations
    """

    cleanedReviewList = [regex_ops(sent) for sent in reviewList]
    return cleanedReviewList
