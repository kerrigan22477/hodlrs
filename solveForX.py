import numpy as np
from numpy.linalg import inv
from tools import Tools

class SolveForX:
    def __init__(self):
        self.Ks = {}
        self.x = []

    def sherWood(self, U, Vt, I):
        # i8 + K1i @ (K2i @ K0)
        # x = b - (U @ inv(I + Vt@U) @ Vt @ b)
        inverse = I - (U @ inv(I + Vt @ U) @ Vt)
        # K0 after factoring
        return inverse

    def solveForX(self, b, root, level):
        t = Tools()

        # if a Leaf
        if root.left is None:
            full_matrix = root.value

            # invert
            Kni = inv(full_matrix)

            # update b w/ inverted matrix
            self.x = Kni@b

            update = np.identity(2)
            next_update = Kni

            return self.x, Kni, update, next_update

        # if NOT LEAF
        else:
            b1, b2 = t.splitQ(b)
            x1, upperLeft_old, update_UL, next_update_UL = self.solveForX(b1, root.left, level - 1)
            x2, lowerRight_old, update_LR, next_update_LR = self.solveForX(b2, root.right, level - 1)

            # things for rebuilding
            z = np.zeros((2**level, 2**level))
            I = np.identity(2**(level+1))

            # Rebuild current level matrix
            K1 = root.value
            U1, S1, VT1, U2, S2, VT2 = K1[0], K1[1], K1[2], K1[3], K1[4], K1[5]

            upperRight = (U1.dot(S1)).dot(VT1)
            lowerLeft = (U2.dot(S2)).dot(VT2)

            K1 = t.buildBlock(z, upperRight, lowerLeft, z)

            # Rebuild previous level's factored out/inverted matrix
            prev_K = t.buildBlock(upperLeft_old, z, z, lowerRight_old)

            # rebuild update needed for current matrix w/ previous inverse matrices
            update = t.buildBlock(update_UL, z, z, update_LR)
            next_update = t.buildBlock(next_update_UL, z, z, next_update_LR)

            # U, Vt, I
            K1i = self.sherWood(prev_K, (update@K1), I)

            # update b w/ previous inverse matrices
            self.x = np.concatenate((x1, x2), axis=0)
            self.x = K1i@self.x

            # add newly inverted matrix to next update
            new_next_update = K1i @ next_update

            return self.x, K1i, next_update, new_next_update