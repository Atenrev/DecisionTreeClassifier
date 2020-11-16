import numpy as np
import pandas as pd

training_file = "res/adult.data"
test_file = "res/adult.test"


def write_to_file(tree):
    with open('out/result.txt', 'w', encoding='utf8') as res:
        res.write(str(tree))
        res.close()

def read_dataset(path):
    return pd.read_csv(path, header=0, delimiter=', ')

def preprocessing():
    dataset_train = read_dataset(training_file)
    dataset_test = read_dataset(test_file)

    print('\nDatatypes\n')
    print(dataset_train.dtypes)
    print()

    # Ignorem fnlwgt ja que no afecta a la classificació
    dataset_train.drop("fnlwgt", axis="columns", inplace=True)
    # Ignorem education-num ja que és una representació numèrica d'education (redundant)
    dataset_train.drop("education-num", axis="columns", inplace=True)
    dataset_train.drop("capital-gain", axis="columns", inplace=True)
    dataset_train.drop("capital-loss", axis="columns", inplace=True)
    '''dataset_train['age'] = pd.qcut(dataset_train['age'], q=4, labels=['young', 'middle-young', 'middle-old', 'old'])
    dataset_train['hours-per-week'] = pd.qcut(dataset_train['hours-per-week'], q=2)
    # Els dos següents són candidats de ser eliminats degut al gran nombre de 0s que tenen...
    dataset_train['capital-gain'] = pd.cut(dataset_train['capital-gain'], bins=2)
    dataset_train['capital-loss'] = pd.cut(dataset_train['capital-loss'], bins=2)'''

    # Ignorem fnlwgt ja que no afecta a la classificació
    dataset_test.drop("fnlwgt", axis="columns", inplace=True)
    # Ignorem education-num ja que és una representació numèrica d'education (redundant)
    dataset_test.drop("education-num", axis="columns", inplace=True)

    dataset_test.drop("capital-gain", axis="columns", inplace=True)
    dataset_test.drop("capital-loss", axis="columns", inplace=True)
    '''dataset_test['age'] = pd.qcut(dataset_test['age'], q=4, labels=['young', 'middle-young', 'middle-old', 'old'])
    dataset_test['hours-per-week'] = pd.qcut(dataset_test['hours-per-week'], q=2)
    # Els dos següents són candidats de ser eliminats degut al gran nombre de 0s que tenen...
    dataset_test['capital-gain'] = pd.cut(dataset_test['capital-gain'], bins=2)
    dataset_test['capital-loss'] = pd.cut(dataset_test['capital-loss'], bins=2)'''
    print('\nHead of train dataset\n')
    print(dataset_train.head())
    print()

    x_train = dataset_train.drop("income", axis="columns").to_numpy()
    y_train = dataset_train["income"].to_numpy()
    x_test = dataset_test.drop("income", axis="columns").to_numpy()
    y_test = dataset_test["income"].to_numpy()

    return dataset_train.columns, x_train, y_train, x_test, y_test