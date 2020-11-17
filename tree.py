from utilities import *
from enum import Enum

class Criterion(Enum):
    ID3 = 1
    GINI = 3

class DecisionTreeClassifier:

    def __init__(self, attr_headers, continuous_attr_header, criterion):
        self.criterion = criterion
        self.attr_headers = attr_headers
        self.attribute_values = []
        self.continuous_attr_header = continuous_attr_header

    def set_attribute_values(self, data):
        self.attribute_values = [np.unique(data[:, i]) for i in range(data.shape[1])]

    def set_labels(self, labels):
        self.attribute_values = labels

    def fit(self, X, Y):
        self.model = SubTree(Y, X, self.attr_headers, self.criterion, self.attribute_values, self.continuous_attr_header)

    def predict(self, X):
        return self.model.predict(X)

    def __str__(self):
        return str(self.model)


class SubTree:

    def __init__(self, s, A, A_header, criterion, labels, continuous_attributes_header):
        self.s = s
        self.A = A
        self.A_header = A_header
        self.negative, self.positive = np.unique(self.s)
        self.criterion = criterion
        self.continuous_attributes_header = continuous_attributes_header
        self.labels = labels
        self.continuous_partition = -1
        self.discretized_continuous_column = None
        self.attribute = self.select_attribute()
        self.child_nodes = self.develop_child_nodes()

    def calculate_gain(self, a):
        if self.criterion == Criterion.ID3:
            return gain(self.s, a)
        else:  # GINI
            return gini_gain(self.s, a)

    def attribute_continuous(self, attribute):
        return (self.A_header[attribute] in self.continuous_attributes_header)

    ''' #########################################################################################
    Select the best partition for a continuous attribute.
        ######################################################################################### '''
    def select_partition(self, a, max_gain, max_index, attribute):  #the gain and gini_gain functions have the nan correction implemented
        values = np.unique(a)
        index = np.where(a == '?')

        for i in values[:-1]:
            a_discrete = np.where(a > i, 'major', 'menor')
            a_discrete[index] = '?'

            current_gain = self.calculate_gain(a_discrete)

            if current_gain > max_gain:
                max_gain = current_gain
                max_index = attribute
                self.continuous_partition = i
                self.discretized_continuous_column = a_discrete

        return max_gain, max_index

    ''' #########################################################################################
    Select the best attribute for making the decision on the subtree root (based on the selection criterion).
        ######################################################################################### '''
    def select_attribute(self):  #the gain and gini_gain functions have the nan correction implemented
        max_index = 0
        max_gain = 0

        for attribute in range(len(self.A_header)):
            a = self.A.T[attribute, :]

            if not self.attribute_continuous(attribute):
                current_gain = self.calculate_gain(a)

                if current_gain > max_gain:
                    max_gain = current_gain
                    max_index = attribute
                    self.continuous_partition = -1
            else:
                max_gain, max_index = self.select_partition(a, max_gain, max_index, attribute)

        return max_index


    ''' #########################################################################################
    It calculates the column and the values corresponding to the current attribute.
        ######################################################################################### '''
    def treat_attribute(self):
        if not self.attribute_continuous(self.attribute):
            curr_col = self.A[:, self.attribute]
            labels = self.labels[self.attribute]
            labels = labels[np.where(labels != '?')] #we're not considering nan values
        else:
            labels = ['major', 'menor']

            if self.continuous_partition == -1: #if the gain is 0 (same value for all the samples)
                a = self.A.T[self.attribute, :]
                curr_col = np.where(a < 0, 'major', 'menor')
            else:
                curr_col = self.discretized_continuous_column

        return curr_col, labels

    ''' #########################################################################################
    Find the child nodes of the current decision node (that will be leaves or other decision nodes).
        ######################################################################################### '''
    def develop_child_nodes(self):
        child_nodes = {}
        curr_col, labels = self.treat_attribute()

        if self.A.shape[1] == 1:  # If it's the last attribute to expand, then the child nodes will be leaves (labeled with the majority class on each attribute value)
            self.develop_last_attribute(child_nodes, curr_col, labels)
        else:
            # If not, explore each value of the attribute
            for label in labels:
                index = np.where(curr_col == label)
                sv = self.s[index]
                svu, count = np.unique(sv, return_counts=True)


                if svu.shape[0] == 1:
                    # if all the data corresponding to this attribute value belongs to the same class, the child is a leaf
                    child_nodes[label] = svu[0]
                elif svu.shape[0] == 0:
                    #If there is no data with the corresponding value, the child is a leaf labeled with the majority class on the father
                    self.develop_child_with_no_data(child_nodes, label)
                else:
                    classes = np.unique(self.A[index, 1 - self.attribute])
                    if self.A.shape[1] == 2 and classes.shape[0] == 1:  #special case, if not, on the next generated node every leaf will have the same class
                        child_nodes[label] = svu[np.argmax(count)]
                    else:
                        if self.continuous_partition != -1:  # If the attribute is continuous and it has a positive gain, we don't delete the attribute
                            child_nodes[label] = SubTree(sv, self.A[index],
                                                         self.A_header, self.criterion,
                                                         self.labels,
                                                         self.continuous_attributes_header)
                        else:
                            child_nodes[label] = SubTree(sv, np.delete(self.A[index], self.attribute, 1),
                                                         np.delete(self.A_header, self.attribute), self.criterion,
                                                         [self.labels[i] for i in range(len(self.labels)) if
                                                          i != self.attribute],
                                                         self.continuous_attributes_header)

        return child_nodes

    ''' #########################################################################################
    If it's the last attribute to expand, then the child nodes will be leaves (labeled with the majority class on each attribute value)
        ######################################################################################### '''
    def develop_last_attribute(self, child_nodes, curr_col, labels):
        for label in labels:
            index = np.where(curr_col == label)
            sv = self.s[index]
            classes, count = np.unique(sv, return_counts=True)

            if len(classes) == 0: #No data
                self.develop_child_with_no_data(child_nodes, label)
            elif len(classes) != 1 and len(np.unique(count)) == 1:  # equiprobable case
                child_nodes[label] = classes[0]
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
    It converts the values of a continuous attribute to the values 'major' or 'menor' depending on the partition
    of the attribute.
        ######################################################################################### '''
    def convert_continuous_attribute(self, label):
        if self.continuous_partition == -1: #if the gain was 0, then every sample has the same value
            continuous_partition = self.A.T[self.attribute, :][0]
        else:
            continuous_partition = self.continuous_partition

        if label > continuous_partition:
            label = 'major'
        else:
            label = 'menor'

        return label

    ''' #########################################################################################
    It counts the number of classes on a leaf of the tree, depending if it's the first iteration or not of the 
    predict_single_count function.
        ######################################################################################### '''
    def count_classes(self, first_iteration, child_node, label, n):
        if first_iteration == True:
            curr_col = self.A[:, self.attribute]
            count = len(np.where(curr_col == label)[0])
        else:
            count = n

        if child_node == self.negative:
            return count, 0
        else:
            return 0, count

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
            if self.attribute_continuous(self.attribute):
                label = self.convert_continuous_attribute(label)

            child_node = self.child_nodes[label]

            if type(child_node) is SubTree:
                if self.continuous_partition == -1:
                    return child_node.predict_single_count(np.delete(X, self.attribute), False)
                else:
                    return child_node.predict_single_count(X, False)
            else:
                return self.count_classes(first_iteration, child_node, label, n)


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

            if self.attribute_continuous(self.attribute):
                label = self.convert_continuous_attribute(label)

            child_node = self.child_nodes[label]

            if type(child_node) is SubTree:
                if self.continuous_partition == -1:
                    return child_node.predict_single(np.delete(X, self.attribute))
                else:
                    return child_node.predict_single(X)
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

        i = ''
        if self.attribute_continuous(self.attribute):
            if self.continuous_partition == -1:  # if the gain was 0, then every sample has the same value
                i = ' que ' + str(self.A.T[self.attribute, :][0])
            else:
                i = ' que ' + str(self.continuous_partition)

        for label, child_node in self.child_nodes.items():
            if type(child_node) is SubTree:
                output += '\t' + tabs + 'Value=' + str(label) + i + ': ' + child_node.__str__(level+1)
            else:
                output += '\t' + tabs + 'Value=' + str(label) + i +': class = ' + str(child_node) + '\n'

        return output




    def select_attribute_discrete(self):  #the gain and gini_gain functions have implemented the nan correction
        if self.criterion == Criterion.ID3:
            gain_attributes = np.array([gain(self.s, a) for a in self.A.T])
        else:  #Criterion.GINI
            gain_attributes = np.array([gini_gain(self.s, a) for a in self.A.T])

        return np.argmax(gain_attributes)

    def develop_child_nodes_discrete(self):
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
                    child_nodes[label] = SubTree(sv, np.delete(self.A[index], self.attribute, 1),
                                         np.delete(self.A_header, self.attribute), self.criterion,
                                         [self.labels[i] for i in range(len(self.labels)) if i != self.attribute],
                                         self.continuous_attributes_header)

        return child_nodes


