import numpy as np
import pandas as pd
from tree import DecisionTreeClassifier, Criterion
from preprocessing import preprocessing, write_to_file
from validation import Validation, Measure
from math import log
from random_forest import RandomForest

def main():
    columns, x_train, y_train, x_test, y_test = preprocessing()
    random_forest_ID3 = RandomForest(columns[:-1], ['age', 'hours-per-week', 'capital-gain', 'capital-loss'],
                                                                  Criterion.ID3, np.vstack((x_train, x_test)), 10)
    decision_tree_ID3 = DecisionTreeClassifier(columns[:-1], ['age', 'hours-per-week', 'capital-gain', 'capital-loss'],
                                                             Criterion.ID3)

    random_forest_GINI = RandomForest(columns[:-1], ['age', 'hours-per-week', 'capital-gain', 'capital-loss'],
                                                                  Criterion.GINI, np.vstack((x_train, x_test)), 10)
    decision_tree_GINI = DecisionTreeClassifier(columns[:-1], ['age', 'hours-per-week', 'capital-gain', 'capital-loss'],
                                                             Criterion.GINI)
    decision_tree_ID3.set_attribute_values(np.vstack((x_train, x_test)))
    decision_tree_GINI.set_attribute_values(np.vstack((x_train, x_test)))
    validation = Validation(x_train, y_train, x_test, y_test)

    print('K-fold validation:\n\n')
    print('Criteri ID3:\n')
    print('Random forest:\n')
    score = validation.score_cross_val(3, random_forest_ID3)
    print(f'Accuracy mitjana: {np.array(score[Measure.ACC]).mean()}\n')
    print(f'Specificity mitjana: {np.array(score[Measure.SPEC]).mean()}\n')


    print('Decision tree:\n')
    score = validation.score_cross_val(3, decision_tree_ID3)
    print(f'Accuracy mitjana: {np.array(score[Measure.ACC]).mean()}\n')
    print(f'Specificity mitjana: {np.array(score[Measure.SPEC]).mean()}\n')

    print('Criteri GINI:\n')
    print('Random forest:\n')
    score = validation.score_cross_val(3, random_forest_GINI)
    print(f'Accuracy mitjana: {np.array(score[Measure.ACC]).mean()}\n')
    print(f'Specificity mitjana: {np.array(score[Measure.SPEC]).mean()}\n')


    print('Decision tree:\n')
    score = validation.score_cross_val(3, decision_tree_GINI)
    print(f'Accuracy mitjana: {np.array(score[Measure.ACC]).mean()}\n')
    print(f'Specificity mitjana: {np.array(score[Measure.SPEC]).mean()}\n')

    print('Final model: Random Forest\n')
    print('Resultats finals: \n')
    final_measure = validation.final_measure(random_forest_ID3)
    print(f'Accuracy mitjana: {np.array(final_measure[Measure.ACC]).mean()}\n')
    print(f'Specificity mitjana: {np.array(final_measure[Measure.SPEC]).mean()}\n')


    print('\n\n Exemple d arbre de decisió entrenat amb totes les dades disponible a out/resultat.txt')
    #Imprimim un arbre de decisió entrenat amb totes les dades, per visualitzar, tot i no ser el millor model
    x_data = np.vstack((x_train, x_test))
    y_data = np.hstack((y_train, y_test))
    decision_tree_ID3.fit(x_data, y_data)
    write_to_file(decision_tree_ID3)


if __name__ == "__main__":
    main()
