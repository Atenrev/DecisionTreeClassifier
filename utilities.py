import numpy as np
import pandas as pd
from math import log


''' #########################################################################################
This function calculates the entropy of the set s.
    ######################################################################################### '''
def entropy(s):
    _, counts = np.unique(s, return_counts=True)
    probs = counts / len(s)
    return -sum([prob*log(prob, 2) for prob in probs])

''' #########################################################################################
This function calculates the entropy of the set s condicionated to the attribute a.
    ######################################################################################### '''
def entropy_cond(s, a):
    n_s = len(s)
    labels = np.unique(a)
    entrp = 0

    for label in labels:
        index = np.where(a == label)
        sv = s[index]
        n_sv = len(sv)
        entrp += (n_sv / n_s) * entropy(sv)

    return entrp

''' #########################################################################################
Gain obtained when ramifying using the a attribute.
    ######################################################################################### '''
def gain(s, a):
    non_nan_values = np.where(a != '?')
    a_modified = a[non_nan_values]
    s_modified = s[non_nan_values]

    entrp = entropy(s)
    entrp_cond = entropy_cond(s_modified, a_modified)
    return len(s_modified)/len(s)*(entrp - entrp_cond)

''' #########################################################################################
Gain normalized obtained when ramifying using the a attribute.
    ######################################################################################### '''
def gini(s):
    _, counts = np.unique(s, return_counts=True)
    probs = counts / len(s)
    return 1-sum([prob * prob for prob in probs])

''' #########################################################################################
Gain obtained when ramifying using the a attribute (in terms of Gini index).
    ######################################################################################### '''
def gini_gain(s, a):
    gini_gain = gini(s)

    non_nan_values = np.where(a != '?')
    a_modified = a[non_nan_values]
    s_modified = s[non_nan_values]
    labels = np.unique(a_modified)
    n_s = len(s_modified)

    for label in labels:
        index = np.where(a_modified == label)
        sv = s_modified[index]
        n_sv = len(sv)
        gini_gain -= (n_sv / n_s) * gini(sv)

    return (n_s/len(s))*gini_gain
