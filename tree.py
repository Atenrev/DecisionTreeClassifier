from utilities import *
from enum import Enum

class Criterion(Enum):
    ID3 = 1
    GINI = 3

class DecisionTreeClassifier:

    def __init__(self, attr_headers, class_header, criterion):
        self.criterion = criterion
        self.attr_headers = attr_headers
        self.class_header = class_header
        self.attribute_values = []

    def set_attribute_values(self, data):
        self.attribute_values = [np.unique(data[:, i]) for i in range(data.shape[1])]

    def fit(self, X, Y):
        self.model = SubTree(Y, X, self.class_header, self.attr_headers, self.criterion, self.attribute_values)

    def predict(self, X):
        return self.model.predict(X)

    def __str__(self):
        return str(self.model)


class SubTree:

    def __init__(self, s, A, s_header, A_header, criterion, labels):
        self.s = s
        self.A = A
        self.s_header = s_header
        self.A_header = A_header
        self.negative, self.positive = np.unique(self.s)
        self.criterion = criterion
        self.labels = labels
        self.attribute = self.select_attribute()
        self.child_nodes = self.develop_child_nodes()

    ''' #########################################################################################
    Select the best attribute for making the decision on the subtree root (based on the selection criterion).
        ######################################################################################### '''
    def select_attribute(self):  #the gain and gini_gain functions have implemented the nan correction
        if self.criterion == Criterion.ID3:
            gain_attributes = np.array([gain(self.s, a) for a in self.A.T])
        else:  #Criterion.GINI
            gain_attributes = np.array([gini_gain(self.s, a) for a in self.A.T])

        return np.argmax(gain_attributes)

    ''' #########################################################################################
    Find the child nodes of the current decision node (that will be leaves or other decision nodes).
        ######################################################################################### '''
    def develop_child_nodes(self):
        child_nodes = {}
        curr_col = self.A[:, self.attribute]
        labels = self.labels[self.attribute]
        labels = labels[np.where(labels != '?')] #we're not considering nan values


        if self.A.shape[1] == 1:
        # If it's the last attribute to expand, then the child nodes will be leaves (labeled with the majority class on each attribute value)
            self.develop_last_attribute(child_nodes, curr_col, labels)

        else:
            # If not, explore each value of the attribute
            for label in labels:
                index = np.where(curr_col == label)
                sv = self.s[index]
                svu = np.unique(sv)

                if len(svu) == 1:
                # if all the data corresponding to this attribute value belongs to the same class, the child is a leaf
                    child_nodes[label] = svu[0]
                elif len(svu) == 0:
                #If there is no data with the corresponding value, the child is a leaf labeled with the majority class on the father
                    self.develop_child_with_no_data(child_nodes, label)
                else:
                    child_nodes[label] = SubTree(sv, np.delete(self.A[index], self.attribute, 1), self.s_header,
                            np.delete(self.A_header, self.attribute), self.criterion,
                            [self.labels[i] for i in range(len(self.labels)) if i != self.attribute])

        return child_nodes

    ''' #########################################################################################
    If it's the last attribute to expand, then the child nodes will be leaves (labeled with the majority class on each attribute value)
        ######################################################################################### '''
    def develop_last_attribute(self, child_nodes, curr_col, labels):
        for label in labels:
            index = np.where(curr_col == label)
            sv = self.s[index]
            classes, count = np.unique(sv, return_counts=True)

            if len(classes) != 1 and len(np.unique(count)) == 1:  # equiprobable case
                child_nodes[label] = '?'
            elif len(classes) == 0: #No data
                self.develop_child_with_no_data(child_nodes, label)
            else:
                child_nodes[label] = classes[np.argmax(count)]

    ''' #########################################################################################
    If there is no data to the corresponding attribute value, it will be assigned the majority class on the father.
        ######################################################################################### '''
    def develop_child_with_no_data(self, child_nodes, label):
        classes_father, count_father = np.unique(self.s, return_counts=True)

        if len(classes_father) != 1 and len(np.unique(count_father)) == 1:  # equiprobable case
            child_nodes[label] = classes_father[0] #'?'
        else:
            child_nodes[label] = classes_father[np.argmax(count_father)]

    ''' #########################################################################################
    Returns the number of samples belonging to each class  on de decision nodes and leaves given by the
    set of attributes X. If it's the first time that this function is called (i.e. the current attribute is actually
    NaN), only the cases with the considered value of the current attribute will be counted. If it's called recursively,
    every sample on the final decision node will be counted.
        ######################################################################################### '''
    def predict_single_count(self, X, first_iteration):
        label = X[self.attribute]
        n = self.s.shape[0]

        if label == '?':
            neg, pos = self.predict_nan_value(X)
            return n*neg, n*pos
        else:
            child_node = self.child_nodes[label]

            if type(child_node) is SubTree:
                return child_node.predict_single_count(np.delete(X, self.attribute), False)
            else:
                if first_iteration == True:
                    curr_col = self.A[:, self.attribute]
                    count = len(np.where(curr_col == label)[0])
                else:
                    count = n

                if child_node == self.negative:
                    return count , 0
                else:
                    return 0, count


    ''' #########################################################################################
    Returns the probabilitie of belonging to the positive or negative class for the set of attributes X, 
    where the current attribute is NaN.
        ######################################################################################### '''
    def predict_nan_value(self, X):
        prob_positive = 0
        prob_negative = 0
        labels = self.labels[self.attribute]
        labels = labels[np.where(labels != '?')]
        n = self.s.shape[0]

        for label in labels:
            X[self.attribute] = label
            neg, pos = self.predict_single_count(X, True)
            prob_positive += pos/n
            prob_negative += neg/n

        return prob_negative, prob_positive


    ''' #########################################################################################
    Returns the prediction for the set of attributes X.
        ######################################################################################### '''
    def predict_single(self, X):
        label = X[self.attribute]

        if label == '?':
            prob_negative, prob_positive = self.predict_nan_value(X)

            if prob_positive > prob_negative:
                return self.positive
            else:
                return self.negative
        else:
            child_node = self.child_nodes[label]

            if type(child_node) is SubTree:
                return child_node.predict_single(np.delete(X, self.attribute))
            else:
                return child_node

    ''' #########################################################################################
    Returns the prediction for each set of attributes on the list X.
        ######################################################################################### '''
    def predict(self, X):
        if len(X.shape) == 1:
            return self.predict_single(X)
        else:
            return np.array([self.predict_single(x) for x in X])



    def __str__(self, level=0):
        tabs = level * '\t'
        output = f'Atribut {self.A_header[self.attribute]}:\n'

        for label, child_node in self.child_nodes.items():
            if type(child_node) is SubTree:
                output += '\t' + tabs + 'Value=' + str(label) + ': ' + child_node.__str__(level+1) + '\n'
            else:
                output += '\t' + tabs + 'Value=' + str(label) + ': class = ' + str(child_node) + '\n'

        return output
