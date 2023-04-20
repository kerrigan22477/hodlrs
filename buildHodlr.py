import numpy as np
from BST import BST
from BST import Node
from BST import Leaf
from tools import Tools


class buildHodlr:
    def __init__(self, k, r):
        self.k = k
        self.r = r

    def _buildHodlr(self, bst, parent, child, approx):
        t = Tools()

        # find svd of covariance matrix
        A12, A21, A11, A22 = t.splitMatrix(child)
        U1, S1, VT1 = np.linalg.svd(A12)
        U2, S2, VT2 = np.linalg.svd(A21)

        # add back in 0s to sigma
        S1 = np.diag(S1)
        S2 = np.diag(S2)

        # low rank approximations
        if approx:
            U1 = U1[:, :self.r]
            S1 = S1[0:self.r, :self.r]
            VT1 = VT1[:self.r, :]
            U2 = U2[:, :self.r]
            S2 = S2[0:self.r, :self.r]
            VT2 = VT2[:self.r, :]

        new_parent = Node([U1, S1, VT1, U2, S2, VT2])

        # tree builds left to right, so
        if parent.left is None:
            bst.put(parent, new_parent, "left")
        else:
            bst.put(parent, new_parent, "right")

        # stop dividing when matrix is 2x2
        if len(A11) <= 3:
            # put in final leaves
            bst.put(new_parent, Leaf(A11), "left")
            bst.put(new_parent, Leaf(A22), "right")
            return bst

        self._buildHodlr(bst, new_parent, A11, approx)
        self._buildHodlr(bst, new_parent, A22, approx)

    def buildHodlr(self, covMat, approx):
        t = Tools()

        # put in root
        A12, A21, A11, A22 = t.splitMatrix(covMat)
        U1, S1, VT1 = np.linalg.svd(A12)
        U2, S2, VT2 = np.linalg.svd(A21)

        # add back in 0s
        S1 = np.diag(S1)
        S2 = np.diag(S2)

        if approx:
            # low rank approx
            U1 = U1[:, :self.r]
            S1 = S1[0:self.r, :self.r]
            VT1 = VT1[:self.r, :]
            U2 = U2[:, :self.r]
            S2 = S2[0:self.r, :self.r]
            VT2 = VT2[:self.r, :]

        #set up bst
        root = Node([U1, S1, VT1, U2, S2, VT2])
        bst = BST(root)

        # build recursively
        self._buildHodlr(bst, root, A11, approx)
        self._buildHodlr(bst, root, A22, approx)

        # return bst, root
        return bst, root


