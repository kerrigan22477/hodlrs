import numpy as np
from numpy.linalg import inv
from tools import Tools
from BST import Node, Leaf

class SolveForX:
    def __init__(self):
        self.Ks = {}

    def factor(self, root, level):

        if root.left is not None:
            self.factor(root.left, level + 1)
            self.factor(root.right, level + 1)

            A = root.value
            U1, S1, VT1, U2, S2, VT2 = A[0], A[1], A[2], A[3], A[4], A[5]

            #upperRight = (U1.dot(S1)).dot(VT1)
            #lowerLeft = (U2.dot(S2)).dot(VT2)
            upperRight = [U1, S1, VT1]
            lowerLeft = [U2, S2, VT2]

            try:
                self.Ks[level] = self.Ks[level] + [upperRight, lowerLeft]
            except KeyError:
                self.Ks[level] = [upperRight, lowerLeft]

        else:
            full_matrix = root.value
            try:
                self.Ks[level] = self.Ks[level] + [full_matrix]
            except KeyError:
                self.Ks[level] = [full_matrix]

        return self.Ks