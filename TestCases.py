import unittest
import numpy as np
import pandas as pd
from utilities import entropy, entropy_cond, gini, gini_gain, gain
from tree import DecisionTreeClassifier, Criterion, SubTree
from math import log

class TestCases(unittest.TestCase):
    '''
    El següent conjunt de tests està basat en l'exercici 4 de problemes,
    en el que s'ha calculat un arbre de decisió (amb criteri de selecció ID3) a mà
    '''


    def setUp(self):
        data = [['Si', 'No', 'No', 'No'],
                ['Si', 'No', 'Si', 'No'],
                ['No', 'No', 'No', 'Si'],
                ['No', 'No', 'Si', 'No'],
                ['No', 'Si', 'Si', 'Si']]
        self.df = pd.DataFrame(data, columns=['Operacio major', 'Familia', 'Gran', 'Enviar a casa'])
        self.tree = DecisionTreeClassifier(self.df.columns[:-1], [], Criterion.ID3)
        self.tree.set_attribute_values(self.df.to_numpy()[:, 0:3])

        data_nan = [['Si', 'No', 'No', 'No'],
                ['?', 'No', 'Si', 'No'],
                ['No', 'No', 'No', 'Si'],
                ['No', 'No', '?', 'No'],
                ['No', 'Si', '?', 'Si']]
        self.df_nan = pd.DataFrame(data_nan, columns=['Operacio major', 'Familia', 'Gran', 'Enviar a casa'])


        '''Exemple vist a la diapositiva 76 de teoria (decision trees),  forçem l'arbre que surt en la diapositiva,
        ja que sabem (manualment) que haurien de donar en aquest arbre les prediccioins dels atributs [?, c2], [?, ?]
        '''
        data_2 = [['b1', 'c2', 'Yes'],
                ['b1', 'c2', 'Yes'],
                ['b1', 'c2', 'Yes'],
                ['b1', 'c2', 'Yes'],
                ['b2', 'c1', 'Yes'],
                ['b2', 'c1', 'Yes'],
                ['b2', 'c1', 'Yes'],
                ['b2', 'c2', 'No'],
                ['b2', 'c2', 'No'],
                ['b2', 'c2', 'No'],
                ['b2', 'c2', 'No'],
                ['b2', 'c2', 'No']]
        self.df2 = pd.DataFrame(data_2, columns=['A', 'B', 'Objectiu'])
        self.tree2 = DecisionTreeClassifier(self.df2.columns[:-1], [], Criterion.ID3)
        self.tree2.set_attribute_values(self.df2.to_numpy()[:, 0:2])

    def test_entropy(self):
        s = self.df.values[:, 3]
        A = self.df.values[:, 0:3]
        entrpy = entropy(s)
        entrpy_cond = []
        for a in A.T:
            entrpy_cond.append(entropy_cond(s, a))

        expected_entrpy = -3/5*log(3/5, 2) - 2/5*log(2/5, 2)
        expected_entrpy_cond = []
        expected_entrpy_cond.append(3/5*(-1/3*log(1/3, 2) - 2/3*log(2/3, 2))) #operacio major
        expected_entrpy_cond.append(4/5*(-3/4*log(3/4, 2) - 1/4*log(1/4, 2))) #familia
        expected_entrpy_cond.append(2 / 5 * (-1 / 2 * log(1 / 2, 2) - 1 / 2 * log(1 / 2, 2))
                                    + 3/5*(-2/3*log(2/3, 2) - 1/3*log(1/3, 2)))  # gran

        self.assertTrue(entrpy == expected_entrpy)
        for i in range(3):
            self.assertTrue(entrpy_cond[i] == expected_entrpy_cond[i])

    def test_gini(self):
        s = self.df.values[:, 3]
        A = self.df.values[:, 0:3]
        gin = gini(s)
        gin_gain = []
        for a in A.T:
            gin_gain.append(gini_gain(s, a))

        expected_gini = 1 - 3/5*3/5 - 2/5*2/5
        expected_gini_gain = []
        expected_gini_gain.append(expected_gini - 3/5*(1 - 1/3*1/3 - 2/3*2/3)) #operacio major
        expected_gini_gain.append(expected_gini - 4/5*(1 -3/4*3/4 - 1/4*1/4)) #familia
        expected_gini_gain.append(expected_gini - 2 / 5 * (1 - 1/2*1/2 - 1/2*1/2)
                                    - 3/5*(1 -2/3*2/3 - 1/3*1/3))  # gran

        self.assertTrue(gin == expected_gini)
        for i in range(3):
            self.assertTrue(gin_gain[i] == expected_gini_gain[i])

    def test_tree(self):
        self.tree.fit(self.df.values[:, 0:3], self.df.values[:, 3])
        node0 = self.tree.model
        self.assertTrue(type(node0) == SubTree)

        node1 = node0.child_nodes['No']
        node2 = node0.child_nodes['Si']
        self.assertTrue(type(node1) == SubTree)

        node3 = node1.child_nodes['No']
        node4 = node1.child_nodes['Si']
        self.assertTrue(type(node3) == SubTree)

        node5 = node3.child_nodes['No']
        node6 = node3.child_nodes['Si']
        self.assertTrue(type(node2) != SubTree)
        self.assertTrue(type(node4) != SubTree)
        self.assertTrue(type(node5) != SubTree)
        self.assertTrue(type(node6) != SubTree)

        #decision nodes
        self.assertTrue(node0.A_header[node0.attribute] == 'Operacio major')
        self.assertTrue(node1.A_header[node1.attribute] == 'Familia')
        self.assertTrue(node3.A_header[node3.attribute] == 'Gran')

        #leaves
        self.assertTrue(node2 == 'No')
        self.assertTrue(node4 == 'Si')
        self.assertTrue(node5 == 'Si')
        self.assertTrue(node6 == 'No')

    def test_predict(self):
        self.tree.fit(self.df.values[:, 0:3], self.df.values[:, 3])
        test = pd.DataFrame([['No', 'No', 'No'],
                ['No', 'No', 'Si'],
                ['No', 'Si', 'No'],
                ['No', 'Si', 'Si'],
                ['Si', 'No', 'No'],
                ['Si', 'No', 'Si'],
                ['Si', 'Si', 'No'],
                ['Si', 'Si', 'Si']],
                columns=['Operacio major', 'Familia', 'Gran']).to_numpy()
        output = self.tree.predict(test).tolist()
        expected_output = ['Si', 'No', 'Si', 'Si', 'No', 'No', 'No', 'No']

        self.assertListEqual(output, expected_output)

    def test_nan_gain_entropy(self):
        s = self.df_nan.values[:, 3]
        A = self.df_nan.values[:, 0:3]

        gain_output = []
        for a in A.T:
            gain_output.append(gain(s, a))

        expected_entrpy = -3/5*log(3/5, 2) - 2/5*log(2/5, 2)
        expected_gain = []
        expected_gain.append(4/5*(expected_entrpy - 3 / 4 * (-1 / 3 * log(1 / 3, 2) - 2 / 3 * log(2 / 3, 2))))  # operacio major
        expected_gain.append(expected_entrpy - 4 / 5 * ( - 3 / 4 * log(3 / 4, 2) - 1 / 4 * log(1 / 4, 2)))  # familia
        expected_gain.append(3/5*(expected_entrpy - 2 / 3 * (-1 / 2 * log(1 / 2, 2) - 1 / 2 * log(1 / 2, 2))))  # gran

        for i in range(3):
            self.assertTrue(gain_output[i] == expected_gain[i])

    def test_gini_nan(self):
        s = self.df_nan.values[:, 3]
        A = self.df_nan.values[:, 0:3]
        gin = gini(s)
        gin_gain = []
        for a in A.T:
            gin_gain.append(gini_gain(s, a))

        expected_gini = 1 - 3/5*3/5 - 2/5*2/5
        expected_gini_gain = []
        expected_gini_gain.append(4/5*(expected_gini - 3/4*(1 - 1/3*1/3 - 2/3*2/3))) #operacio major
        expected_gini_gain.append(expected_gini - 4/5*(1 -3/4*3/4 - 1/4*1/4)) #familia
        expected_gini_gain.append(3/5*(expected_gini - 2 / 3 * (1 - 1/2*1/2 - 1/2*1/2)))  # gran

        self.assertTrue(gin == expected_gini)
        for i in range(3):
            self.assertTrue(gin_gain[i] == expected_gini_gain[i])

    def test_predict_nan(self):
        self.tree2.fit(self.df2.values[:, 0:2], self.df2.values[:, 2])
        test = pd.DataFrame([['?', 'c2'],
                ['?', '?']],
                columns=['B', 'C']).to_numpy()
        output = self.tree2.predict(test).tolist()
        expected_output = ['No', 'Yes']

        probabilities1 = self.tree2.model.predict_nan_value(['?', 'c2'])
        probabilities2 = self.tree2.model.predict_nan_value(['?', '?'])

        expected_probabilities1 = 8/12, 4/12
        expected_probabilities2 = (8/12)*(5/8), (4/12 + (8/12)*(3/8))

        self.assertListEqual(output, expected_output)
        self.assertTupleEqual(probabilities1, expected_probabilities1)
        self.assertTrue(abs(probabilities2[0] - expected_probabilities2[0])< 1E-15) #truncation error
        self.assertEqual(probabilities2[1], probabilities2[1])