from numpy.linalg import inv
import numpy as np
from tools import Tools
from BST import BST
from BST import Node
from BST import Leaf

# Convert invert so it produces a HODLR matrix

class Invert:
    def __init__(self):
        self.u = []
        self.zeros = []
        self.r = 16

    def invMath(self, x11, x22, vt1, vt2, u1, u2, S1, S2):
        t = Tools()

        Y = t.buildBlock((vt2.dot(x11)).dot(u1), inv(S1), inv(S2), (vt1.dot(x22)).dot(u2))
        Y = inv(Y)

        self.zeros = np.zeros((len(x11), len(x11[0])))
        a = t.buildBlock(x11, self.zeros, self.zeros, x22)

        self.zeros = np.zeros((len(x11.dot(u1)), len((x11.dot(u1))[0])))
        b = t.buildBlock(x11.dot(u1), self.zeros, self.zeros, x22.dot(u2))

        self.zeros = np.zeros((len(vt1.dot(x11)), len((vt1.dot(x11))[0])))
        d = t.buildBlock(vt2.dot(x11), self.zeros, self.zeros, vt1.dot(x22))

        C = a - b@(Y@d)

        return C

    def invert(self, root):
        A = root.value
        t = Tools()

        U1, S1, VT1, U2, S2, VT2 = A[0], A[1], A[2], A[3], A[4], A[5]
        A12, A21, A11, A22 = t.splitA(A, root)

        if root.left.left is None:
            # if Leaf, invert normally
            X11 = inv(A11)
            X22 = inv(A22)

        else:
            # if not leaf
            X11 = self.invert(root.left)
            X22 = self.invert(root.right)

        # invert non-leaves
        C = self.invMath(X11, X22, VT1, VT2, U1, U2, S1, S2)
        # print(C)
        return C