import numpy as np
from split import Split
from BST import BST
from BST import Node
from BST import Leaf
from tools import Tools


class buildHodlr:
    def __init__(self):
        self.clusters = []
        self.data = []
        self.k = 0
        self.centroids = []
        self.r = 0
        self.finalPoints = []

    def _buildHodlr(self, bst, parent, child, k, approx):
        # if size of matrix is uneven, make even w/0s so it splits properly
        t = Tools()

        # find svd of covariance matrix
        s = Split()
        A11, A12, A21, A22 = s.split(child)
        U1, S1, VT1 = np.linalg.svd(A12)
        U2, S2, VT2 = np.linalg.svd(A21)

        # low rank approximations
        # add back in 0s to sigma
        S1 = np.diag(S1)
        S2 = np.diag(S2)

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

        #if len(A11) < 2*self.k:
        if len(A11) <= 3:
            # put in final leaves
            bst.put(new_parent, Leaf(A11), "left")
            bst.put(new_parent, Leaf(A22), "right")
            return bst

        # (self, bst, parent, child, c1_points, c2_points, k, approx)
        self._buildHodlr(bst, new_parent, A11, k, approx)
        self._buildHodlr(bst, new_parent, A22, k, approx)

    def buildHodlr(self, data, k, r, covMat, approx):
        t = Tools()
        self.data = data
        self.k = k
        self.r = r

        # put in root
        s = Split()
        A11, A12, A21, A22 = s.split(covMat)
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
        self._buildHodlr(bst, root, A11, k, approx)
        self._buildHodlr(bst, root, A22, k, approx)


        # return bst, root
        return bst, root, self.finalPoints


