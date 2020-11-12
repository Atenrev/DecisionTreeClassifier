import numpy as np
import pandas as pd
from tree import DecisionTreeClassifier
from math import log

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,  precision_score, recall_score, roc_curve,roc_auc_score, auc
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, precision_recall_curve

training_file = "res/adult.data"
test_file = "res/adult.test"


def read_dataset(path):
    return pd.read_csv('res/adult.data', header=0, delimiter=', ')


def write_to_file(tree):
    with open('out/result.txt', 'w', encoding='utf8') as res:
        res.write(str(tree))
        res.close()


def main():
    dataset_train = read_dataset('res/adult.data')
    dataset_test = read_dataset('res/adult.test')

    print('\nDatatypes\n')
    print(dataset_train.dtypes)
    print()

    # Ignorem fnlwgt ja que no afecta a la classificació
    dataset_train.drop("fnlwgt", axis="columns", inplace=True)
    # Ignorem fnlwgt ja que és una representació numèrica d'education (redundant)
    dataset_train.drop("education-num", axis="columns", inplace=True)

    dataset_train['age'] = pd.qcut(dataset_train['age'], q=4, labels=['young', 'middle-young', 'middle-old', 'old'])
    dataset_train['hours-per-week'] = pd.qcut(dataset_train['hours-per-week'], q=2)
    # Els dos següents són candidats de ser eliminats degut al gran nombre de 0s que tenen...
    dataset_train['capital-gain'] = pd.cut(dataset_train['capital-gain'], bins=2)
    dataset_train['capital-loss'] = pd.cut(dataset_train['capital-loss'], bins=2)

    # Ignorem fnlwgt ja que no afecta a la classificació
    dataset_test.drop("fnlwgt", axis="columns", inplace=True)
    # Ignorem fnlwgt ja que és una representació numèrica d'education (redundant)
    dataset_test.drop("education-num", axis="columns", inplace=True)

    dataset_test['age'] = pd.qcut(dataset_test['age'], q=4, labels=['young', 'middle-young', 'middle-old', 'old'])
    dataset_test['hours-per-week'] = pd.qcut(dataset_test['hours-per-week'], q=2)
    # Els dos següents són candidats de ser eliminats degut al gran nombre de 0s que tenen...
    dataset_test['capital-gain'] = pd.cut(dataset_test['capital-gain'], bins=2)
    dataset_test['capital-loss'] = pd.cut(dataset_test['capital-loss'], bins=2)

    print('\nHead of train dataset\n')
    print(dataset_train.head())
    print()
    
    data_train = dataset_train.values
    data_test = dataset_train.values

    x_train = dataset_train.drop("income", axis="columns").to_numpy()
    y_train = dataset_train["income"].to_numpy()
    x_test = dataset_test.drop("income", axis="columns").to_numpy()
    y_test = dataset_test["income"].to_numpy()

    model = DecisionTreeClassifier(dataset_train.columns[:-1], dataset_train.columns[-1])
    model.fit(x_train, y_train)
    # write_to_file(model)
    print(f'\nPrediction for first test sample: {model.predict(x_test[0])}\n')
    predictions = model.predict(x_test)

    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions, average='micro')
    rec = recall_score(y_test, predictions, average='micro')
    print(f'Accuracy:{acc}')
    print(f'Precision:{prec}')
    print(f'Recall:{rec}')

    error = plot_confusion_matrix(model, x_test, y_test, cmap=plt.cm.Blues)
    error = 1-(sum(np.diag(error.confusion_matrix)) / sum(error.confusion_matrix.ravel()))
    print(f"La taxa d'error és de: {error*100}%")


if __name__ == "__main__":
    main()
