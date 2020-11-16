import numpy as np
import pandas as pd
from tree import SubTree, DecisionTreeClassifier
from enum import Enum


class Measure(Enum):
    ACC = 1
    PR = 2
    F1 = 3
    REC = 4
    SPEC = 5

class Validation:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.negative, self.positive = np.unique(self.y_train)
        self.tp = self.fp = self.tn = self.fn = 0
        self.empty_measures = {Measure.ACC: [], Measure.PR: [], Measure.REC: [], Measure.SPEC: [], Measure.F1: []}

    ''' #########################################################################################
    Given a set of predictions and test data, it compares the sets to count the number of true positives, 
    false positives, true negatives and false negatives on the predictions.
        ######################################################################################### '''
    def count_results(self, predictions, test):
        self.tp = self.fp = self.tn = self.fn = 0

        for k in range(predictions.shape[0]):
            if predictions[k] == self.positive:
                if test[k] == self.positive:
                    self.tp += 1
                else:
                    self.fp += 1
            else:
                if test[k] == self.negative:
                    self.tn += 1
                else:
                    self.fn += 1

    ''' #########################################################################################
    Given a set of predictions and test data, for each measure.
        ######################################################################################### '''
    def calculate_measures(self, measures, predictions, test):
        self.count_results(predictions, test)

        measures[Measure.ACC].append(self.accuracy())
        measures[Measure.PR].append(self.precision())
        measures[Measure.REC].append(self.recall())
        measures[Measure.SPEC].append(self.specificity())
        measures[Measure.F1].append(self.f1_score())

    ''' #########################################################################################
    It calculates the accuracy from the tp, tn, fp and fn obtained on the predictions.
        ######################################################################################### '''
    def accuracy(self):
        return (self.tp + self.tn)/(self.tp + self.tn + self.fp + self.fn)

    ''' #########################################################################################
    It calculates the precision from the tp, tn, fp and fn obtained on the predictions.
        ######################################################################################### '''
    def precision(self):
        return self.tp/(self.tp + self.fp)

    ''' #########################################################################################
    It calculates the recall from the tp, tn, fp and fn obtained on the predictions.
        ######################################################################################### '''
    def recall(self):
        return self.tp/(self.tp + self.fn)

    ''' #########################################################################################
    It calculates the specificity from the tp, tn, fp and fn obtained on the predictions.
        ######################################################################################### '''
    def specificity(self):
        return self.tn/(self.fp + self.tn)

    ''' #########################################################################################
    It calculates the F1-score from the tp, tn, fp and fn obtained on the predictions.
        ######################################################################################### '''
    def f1_score(self):
        precision = self.precision()
        recall = self.recall()

        return precision*recall/(precision + recall)

    ''' #########################################################################################
    Given an integer k, it randomly splits the data in order to generate the k folds for the k-fold validation.
        ######################################################################################### '''
    def generate_folds(self, k):
        x_folds = []
        y_folds = []
        index = np.array(range(self.x_train.shape[0]))
        np.random.seed(1)
        np.random.shuffle(index)
        folds_index = np.array_split(index, k)

        for fold in folds_index:
            x_folds.append(self.x_train[fold])
            y_folds.append(self.y_train[fold])

        return x_folds, y_folds

    ''' #########################################################################################
    Given an integer k and a measure, it applies the k-fold validation to calculate the mean of the measures
    obtained with each fold.
        ######################################################################################### '''
    def score_cross_val(self, k, model):
        x_folds, y_folds = self.generate_folds(k)
        measures = self.empty_measures

        for fold in range(k):
            x_val = x_folds[fold]
            y_val = y_folds[fold]
            x_train = np.vstack(tuple(x_folds[i] for i in range(k) if i != fold))
            y_train = np.hstack(tuple(y_folds[i] for i in range(k) if i != fold))

            model.fit(x_train, y_train)
            predictions = model.predict(x_val)


            self.calculate_measures(measures, predictions, y_val)

        return measures

    ''' #########################################################################################
    It trains a model trained with all the training set and calculates the given measure for the predictions made with 
    the test set. Afterwards, it trains the final model (with all the data) and prints the resulting decision tree.
        ######################################################################################### '''
    def final_measure(self, model):
        model.fit(self.x_train, self.y_train)
        predictions = model.predict(self.x_test)
        measures = self.empty_measures
        self.calculate_measures(measures, predictions, self.y_test)

        return measures




