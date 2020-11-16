import numpy as np
import pandas as pd
from tree import DecisionTreeClassifier, Criterion
from preprocessing import preprocessing, write_to_file
from validation import Validation, Measure
from math import log


def main():
    columns, x_train, y_train, x_test, y_test = preprocessing()
    model = DecisionTreeClassifier(columns[:-1], ['age', 'hours-per-week', 'capital-gain', 'capital-loss'],
                                   Criterion.GINI)
    model.set_attribute_values(np.vstack((x_train, x_test)))
    validation = Validation(x_train, y_train, x_test, y_test)

    m = validation.score_cross_val(3, model)
    print(m)
    #x_data = np.vstack((x_train, x_test))
    #y_data = np.hstack((y_train, y_test))
    #model.fit(x_data, y_data)
    #write_to_file(model)
'''
print(score)
 print(np.array(score[Measure.ACC]).mean())
 print(np.array(score[Measure.REC]).mean())
 #print(score) 
#Final model, trained with all the data
x_data = np.vstack(x_train, x_test)
y_data = np.hstack(y_train, y_test)
model.fit(x_data, y_data)
write_to_file(model)'''



if __name__ == "__main__":
    main()
