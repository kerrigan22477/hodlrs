import numpy as np
from numpy.linalg import inv
from tools import Tools
from BST import Node, Leaf

class SolveForX:
    def __init__(self):
        self.Ks = {}
        self.b = []

    def sherWood(self, U, Vt, I):
        # i8 + K1i @ (K2i @ K0)
        # x = b - (U @ inv(I + Vt@U) @ Vt @ b)
        inverse = I - (U @ inv(I + Vt @ U) @ Vt)
        # K0 after factoring
        return inverse

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

    def solveForX_bruteForce(self, covMat, b, root):
        t = Tools()
        self.factor(root, 0)
        levels = self.Ks

        # rebuild K2
        z2 = np.zeros((2, 2))
        z4 = np.zeros((4, 4))
        i2 = np.identity(2)
        i8 = np.identity(8)

        a = t.buildBlock(levels[2][0], z2, z2, levels[2][1])
        d = t.buildBlock(levels[2][2], z2, z2, levels[2][3])

        K2 = t.buildBlock(a, z4, z4, d)

        b = np.arange(len(covMat)).T

        # invert and solve
        K2i = inv(K2)
        b2 = K2i @ b

        #rebuild K1
        U1 = levels[1][0][0]
        S1 = levels[1][0][1]
        VT1 = levels[1][0][2]
        uppR = (U1.dot(S1)).dot(VT1)

        U2 = levels[1][1][0]
        S2 = levels[1][1][1]
        VT2 = levels[1][1][2]
        lowL = (U2.dot(S2)).dot(VT2)

        U12 = levels[1][2][0]
        S12 = levels[1][2][1]
        VT12 = levels[1][2][2]
        uppR2 = (U12.dot(S12)).dot(VT12)

        U22 = levels[1][3][0]
        S22 = levels[1][3][1]
        VT22 = levels[1][3][2]
        lowL2 = (U22.dot(S22)).dot(VT22)

        a = t.buildBlock(z2, uppR, lowL, z2)
        d = t.buildBlock(z2, uppR2, lowL2, z2)

        K1 = t.buildBlock(a, z4, z4, d)

        # invert and solve
        # U, Vt, b, I
        K1i = self.sherWood(K2i, K1, i8)
        b1 = K1i @ b2

        # rebuild K0 from dict of levels
        U1 = levels[0][0][0]
        S1 = levels[0][0][1]
        VT1 = levels[0][0][2]
        uppR = (U1.dot(S1)).dot(VT1)

        U2 = levels[0][1][0]
        S2 = levels[0][1][1]
        VT2 = levels[0][1][2]
        lowL = (U2.dot(S2)).dot(VT2)

        K0 = t.buildBlock(z4, uppR, lowL, z4)

        # invert and solve
        # multiply by inverse of K2 as if we had factored K2 out
        # K0 needs to have K2 factored out before we factor out K1, hence Vt = K2i@K0
        # U, Vt, I
        K0i = self.sherWood(K1i, K2i @ K0, i8)

        b0 = K0i @ b1

        return b0

    def solveForX(self, covMat, b, root, level):
        t = Tools()


        # PART 2 aka the nodes is a mess fix this later
        # if NOT a Leaf
        if root.left is not None:
            K = root.value
            U1, S1, VT1, U2, S2, VT2 = K[0], K[1], K[2], K[3], K[4], K[5]

            upperRight = (U1.dot(S1)).dot(VT1)
            lowerLeft = (U2.dot(S2)).dot(VT2)

            z = np.zeros((4,4))

            K = t.buildBlock(z, upperRight, lowerLeft, z)

            I = np.identity((8,8))
            # U, Vt, b, I
            # K1_after_factoring = i8 + K2i@K1
            Kii = self.sherWood(K1i, prev_K @ K0, I)
            # print((test_K0i - K0i).round(2))

            upperRight, ur_leftover = self.solveForX(root.left, level + 1)
            lowerLeft, ll_leftover = self.solveForX(root.right, level + 1)

            z2 = np.zeros((2,2))
            prev_K = t.buildBlock(z2, ur_leftover, ll_leftover, z2)


            upperRight = (U1.dot(S1)).dot(VT1)
            lowerLeft = (U2.dot(S2)).dot(VT2)

            upperLeft = self.multiply(root.left, q1)
            lowerRight = self.multiply(root.right, q2)

            u1 = upperRight + A12.dot(q2)
            u2 = lowerLeft + A21.dot(q1)
            self.u = np.concatenate((u1, u2), axis=0)

            return self.u, Kii

        # if LEAF
        else:
            full_matrix = root.value
            # invert and solve
            Kni = inv(full_matrix)

            # update b using previous multiply function (aka do Kni@b)
            K12, K21, K11, K22 = t.splitAi(Kni)
            b1, b2 = t.splitQ(b)

            b1 = K11.dot(b1) + K12.dot(b2)
            b2 = K22.dot(b2) + K21.dot(b1)
            self.b = np.concatenate((b1, b2), axis=0)
            return self.b, Kni


