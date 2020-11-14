import numpy as np
import pandas as pd
from math import log



def entropy(x):
    _, counts = np.unique(x, return_counts=True)
    probs = counts / len(x)
    return -sum([prob*log(prob, 2) for prob in probs])


def entropy_cond(s, x):
    n_s = len(s)
    labels = np.unique(x)
    etpy = 0

    for label in labels:
        index = np.where(x == label)
        sv = s[index]
        n_sv = len(sv)
        etpy += (n_sv / n_s) * entropy(sv)

    return etpy


def gain(s, x):
    es = entropy(s)
    esa = entropy_cond(s, x)
    return es - esa


def split_info(s, x):
    n_s = len(s)
    labels = np.unique(x)
    s_i = 0

    for label in labels:
        index = np.where(x == label)
        sv = s[index]
        n_sv = len(sv)
        frac = n_sv / n_s
        s_i += frac * log(frac, 2)

    return s_i


def gain_ratio(s, x):
    g = gain(s, x)
    s = split_info(s, x)
    return g/s



class DecisionTreeClassifier:

    def __init__(self, attr_headers, class_header, criterion): 
        self.criterion = criterion
        self.attr_headers = attr_headers
        self.class_header = class_header

    def fit(self, X, Y):
        if self.criterion == 'entropy':
            self.model = ID3Tree(Y, X, self.class_header, self.attr_headers)
        else:
            self.model = C45Tree(Y, X, self.class_header, self.attr_headers)

    def predict(self, X):
        return self.model.predict(X)

    def __str__(self):
        return str(self.model)


class ID3Tree:

    def __init__(self, s, A, sh, Ah):
        self.s = s
        self.A = A
        self.sh = sh
        self.Ah = Ah
        self.attribute = self.select_attribute()
        self.leafs = self.develop_leafs()

    def select_attribute(self):
        max_index = index = max_gain = 0

        for a in self.A.T:
            g = gain(self.s, a)
            if g > max_gain:
                max_index = index
                max_gain = g
            index += 1

        return max_index

    def develop_leafs(self):
        leafs = []
        curr_col = self.A[:, self.attribute]
        labels = np.unique(curr_col)

        if self.A.shape[1] == 1:
            # If it's the last node then the leafs will be the classes
            for label in labels:
                index = np.where(curr_col == label)
                sv = self.s[index]
                l, c = np.unique(sv, return_counts=True)

                if len(np.unique(c)) == 1 and len(l) != 1:
                    leafs.append(f'Per al valor {label}, classe ?')
                else:
                    leafs.append(f'Per al valor {label}, classe {l[np.argmax(c)]}')

        else:
            # If not, add the next nodes
            for label in labels:
                index = np.where(curr_col == label)
                sv = self.s[index]
                svu = np.unique(sv)
                if len(svu) == 1:
                    # Except for the equiprobable
                    l, c = np.unique(sv, return_counts=True)
                    leafs.append(f'Per al valor {label}, classe {l[np.argmax(c)]}')
                else:
                    leafs.append(
                        ID3Tree(
                            sv,
                            np.delete(self.A[index], self.attribute, 1),
                            self.sh,
                            np.delete(self.Ah, self.attribute),
                        )
                    )

        return leafs

    def predict_single(self, X):
        curr_col = self.A[:, self.attribute]
        labels = np.unique(curr_col)
        attr = X[self.attribute]
        label_index = np.where(labels == attr)[0][0]
        leaf = self.leafs[label_index]

        if type(leaf) is ID3Tree:
            return leaf.predict_single(np.delete(X, self.attribute))
        else:
            return leaf

    def predict(self, X):
        if len(X.shape) == 1:
            return np.array([self.predict_single(X)])
        
        return np.array([self.predict_single(x) for x in X])



    def __str__(self, level=0):
        tabs = level * '\t'
        output = tabs + f'Atribut {self.Ah[self.attribute]}:\n'

        for leaf in self.leafs:
            if type(leaf) is ID3Tree:
                output += leaf.__str__(level+1)
            else:
                output += '\t' + tabs + str(leaf) + '\n'

        return output





class C45Tree:

    def __init__(self, s, A, sh, Ah):
        self.s = s
        self.A = A
        self.sh = sh
        self.Ah = Ah
        self.attribute = self.select_attribute()
        self.leafs = self.develop_leafs()

    def select_attribute(self):
        max_index = index = max_gain = 0

        for a in self.A.T:
            g = gain_ratio(self.s, a)
            if g > max_gain:
                max_index = index
                max_gain = g
            index += 1

        return max_index

    def develop_leafs(self):
        print("HOLAAA - C4.5")
        leafs = []
        curr_col = self.A[:, self.attribute]
        labels = np.unique(curr_col)

        if self.A.shape[1] == 1:
            # If it's the last node then the leafs will be the classes
            for label in labels:
                index = np.where(curr_col == label)
                sv = self.s[index]
                l, c = np.unique(sv, return_counts=True)

                if len(np.unique(c)) == 1 and len(l) != 1:
                    leafs.append(f'Per al valor {label}, classe ?')
                else:
                    leafs.append(f'Per al valor {label}, classe {l[np.argmax(c)]}')

        else:
            # If not, add the next nodes
            for label in labels:
                index = np.where(curr_col == label)
                sv = self.s[index]
                svu = np.unique(sv)
                if len(svu) == 1:
                    # Except for the equiprobable
                    l, c = np.unique(sv, return_counts=True)
                    leafs.append(f'Per al valor {label}, classe {l[np.argmax(c)]}')
                else:
                    leafs.append(
                        ID3Tree(
                            sv,
                            np.delete(self.A[index], self.attribute, 1),
                            self.sh,
                            np.delete(self.Ah, self.attribute),
                        )
                    )

        return leafs

    def predict_single(self, X):
        curr_col = self.A[:, self.attribute]
        labels = np.unique(curr_col)
        attr = X[self.attribute]
        label_index = np.where(labels == attr)[0][0]
        leaf = self.leafs[label_index]

        if type(leaf) is ID3Tree:
            return leaf.predict_single(np.delete(X, self.attribute))
        else:
            return leaf

    def predict(self, X):
        if len(X.shape) == 1:
            return np.array([self.predict_single(X)])
        
        return np.array([self.predict_single(x) for x in X])



    def __str__(self, level=0):
        tabs = level * '\t'
        output = tabs + f'Atribut {self.Ah[self.attribute]}:\n'

        for leaf in self.leafs:
            if type(leaf) is ID3Tree:
                output += leaf.__str__(level+1)
            else:
                output += '\t' + tabs + str(leaf) + '\n'

        return output
