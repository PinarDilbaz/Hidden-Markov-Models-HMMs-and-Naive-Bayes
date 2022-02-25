import numpy as np


def vocabulary(data):
    """
    Creates the vocabulary from the data.
    :param data: List of lists, every list inside it contains words in that sentence.
                 len(data) is the number of examples in the data.
    :return: Set of words in the data
    """
    size_of_data = len(data)
    vocab = set()
    for i in range(size_of_data):
        for d in data[i]:
            vocab.add(d)
    return vocab

def estimate_pi(train_labels):
    """
    Estimates the probability of every class label that occurs in train_labels.
    :param train_labels: List of class names. len(train_labels) is the number of examples in the training data.
    :return: pi. pi is a dictionary. Its keys are class names and values are their probabilities.
    """
    pi = dict()
    for train_label in train_labels:
        if train_label in pi:
            pi[train_label] = pi[train_label] + 1
        else:
            pi[train_label] = 1

    for p in pi:
        val = pi[p]/len(train_labels)
        pi[p] = val
    return pi
    
def estimate_theta(train_data, train_labels, vocab):
    """
    Estimates the probability of a specific word given class label using additive smoothing with smoothing constant 1.
    :param train_data: List of lists, every list inside it contains words in that sentence.
                       len(train_data) is the number of examples in the training data.
    :param train_labels: List of class names. len(train_labels) is the number of examples in the training data.
    :param vocab: Set of words in the training set.
    :return: theta. theta is a dictionary of dictionaries. At the first level, the keys are the class names. At the
             second level, the keys are all the words in vocab and the values are their estimated probabilities given
             the first level class name.
    """

    theta = dict()
    j = -1
    for train_label in train_labels:
        j = j+1 
        index = []
        summ = list()
        if train_label not in theta.keys():
            theta[train_label] = dict()
            for i in range(len(train_labels)):
                if train_labels[j] == train_labels[i]:
                    index.append(i)
            for t in range(len(index)):
                summ = summ + train_data[index[t]]
            val = len(summ)+len(vocab)
            for v in vocab:
                x = summ.count(v) + 1  
                value = x / val
                theta[train_label][v] = value
    return theta
    

def test(theta, pi, vocab, test_data):
    """
    Calculates the scores of a test data given a class for each class. Skips the words that are not occurring in the
    vocabulary.
    :param theta: A dictionary of dictionaries. At the first level, the keys are the class names. At the second level,
                  the keys are all of the words in vocab and the values are their estimated probabilities.
    :param pi: A dictionary. Its keys are class names and values are their probabilities.
    :param vocab: Set of words in the training set.
    :param test_data: List of lists, every list inside it contains words in that sentence.
                      len(test_data) is the number of examples in the test data.
    :return: scores, list of lists. len(scores) is the number of examples in the test set. Every inner list contains
             tuples where the first element is the score and the second element is the class name.
    """
    
    scores = []
    number_of_data = len(test_data)
    number_of_pi = len(pi)
    number_of_vocab = len(vocab)
    
    for i in range(number_of_data):
        scores.append([])
        for p in pi:
            summ = 0
            for v in vocab:
                value = np.log(theta[p][v]) * test_data[i].count(v)
                summ = summ + value
            x = np.log(pi[p]) + summ
            scores[i].append((x, p))

    return scores
if __name__ == "__main__":
    """
    train_data = open('C:/Users/ASUS/Desktop/Cng409/hw4/nb_data/train_set.txt', 'r')
    test_data = open('C:/Users/ASUS/Desktop/Cng409/hw4/nb_data/test_set.txt', 'r')
    train_labels = open('C:/Users/ASUS/Desktop/Cng409/hw4/nb_data/train_labels.txt', 'r')
    test_labels = open('C:/Users/ASUS/Desktop/Cng409/hw4/nb_data/test_labels.txt', 'r')
"""
