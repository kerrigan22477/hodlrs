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

    def factor(self, root, level):

        if root.left is not None:
            self.factor(root.left, level + 1)
            self.factor(root.right, level + 1)

            A = root.value
            U1, S1, VT1, U2, S2, VT2 = A[0], A[1], A[2], A[3], A[4], A[5]

            uR = (U1.dot(S1)).dot(VT1)
            lL = (U2.dot(S2)).dot(VT2)
            # upperRight = [U1, S1, VT1]
            # lowerLeft = [U2, S2, VT2]

            try:
                # self.Ks[level] = self.Ks[level] + [upperRight, lowerLeft]
                self.Ks[level] = self.Ks[level] + [uR, lL]
            except KeyError:
                # self.Ks[level] = [upperRight, lowerLeft]
                self.Ks[level] = [uR, lL]

        else:
            full_matrix = root.value
            try:
                self.Ks[level] = self.Ks[level] + [full_matrix]
            except KeyError:
                self.Ks[level] = [full_matrix]

    def undoSVD(self, level):
        matrices = []
        for i in range(len(self.Ks[level])):
            U1 = self.Ks[level][i][0]
            S1 = self.Ks[level][i][1]
            VT1 = self.Ks[level][i][2]
            matrices.append(U1.dot(S1).dot(VT1))
        return matrices

    def solveForX16(self, n, b, root, x, covMat):
        t = Tools()
        self.factor(root, 0)

        levels = self.Ks

        '''for x in range(len(levels[2])):
            for i in range(len(levels[2][0])):
                print((levels[2][x][i]).round(1))'''

        z2 = np.zeros((2, 2))
        z4 = np.zeros((4, 4))
        z8 = np.zeros((8, 8))
        z16 = np.zeros((16, 16))
        i32 = np.identity(32)

        # rebuild K4
        a = t.buildBlock(levels[3][0], z2, z2, levels[3][1])
        b = t.buildBlock(levels[3][2], z2, z2, levels[3][3])
        c = t.buildBlock(levels[3][4], z2, z2, levels[3][5])
        d = t.buildBlock(levels[3][6], z2, z2, levels[3][7])
        e = t.buildBlock(levels[3][8], z2, z2, levels[3][9])
        f = t.buildBlock(levels[3][10], z2, z2, levels[3][11])
        g = t.buildBlock(levels[3][12], z2, z2, levels[3][13])
        h = t.buildBlock(levels[3][14], z2, z2, levels[3][15])

        K4_1 = t.buildBlock(a, z4, z4, b)
        K4_2 = t.buildBlock(c, z4, z4, d)
        K4_3  = t.buildBlock(e, z4, z4, f)
        K4_4 = t.buildBlock(g, z4, z4, h)

        K4_UL = t.buildBlock(K4_1, z8, z8, K4_2)
        K4_LR = t.buildBlock(K4_3, z8, z8, K4_4)

        K4 = t.buildBlock(K4_UL, z16, z16, K4_LR)

        # rebuild K3
        a = t.buildBlock(z2, levels[3][0], levels[3][1], z2)
        b = t.buildBlock(z2, levels[3][2], levels[3][3], z2)
        c = t.buildBlock(z2, levels[3][4], levels[3][5], z2)
        d = t.buildBlock(z2, levels[3][6], levels[3][7], z2)
        e = t.buildBlock(z2, levels[3][8], levels[3][9], z2)
        f = t.buildBlock(z2, levels[3][10], levels[3][11], z2)
        g = t.buildBlock(z2, levels[3][12], levels[3][13], z2)
        h = t.buildBlock(z2, levels[3][14], levels[3][15], z2)

        K3_1 = t.buildBlock(a, z4, z4, b)
        K3_2 = t.buildBlock(c, z4, z4, d)
        K3_3  = t.buildBlock(e, z4, z4, f)
        K3_4 = t.buildBlock(g, z4, z4, h)

        K3_UL = t.buildBlock(K3_1, z8, z8, K3_2)
        K3_LR = t.buildBlock(K3_3, z8, z8, K3_4)

        K3 = t.buildBlock(K3_UL, z16, z16, K3_LR)

        # rebuild K2
        #K2_ms = self.undoSVD(2)
        K2_ms = self.Ks[2]

        a = t.buildBlock(z4, K2_ms[0], K2_ms[1], z4)
        b = t.buildBlock(z4, K2_ms[2], K2_ms[3], z4)
        c = t.buildBlock(z4, K2_ms[4], K2_ms[5], z4)
        d = t.buildBlock(z4, K2_ms[6], K2_ms[7], z4)

        K2_UL = t.buildBlock(a, z8, z8, b)
        K2_LR = t.buildBlock(c, z8, z8, d)

        K2 = t.buildBlock(K2_UL, z16, z16, K2_LR)

        # rebuild K1
        #K1_ms = self.undoSVD(1)
        K1_ms = self.Ks[1]

        a = t.buildBlock(z8, K1_ms[0], K1_ms[1], z8)
        d = t.buildBlock(z8, K1_ms[2], K1_ms[3], z8)

        K1 = t.buildBlock(a, z16, z16, d)

        # rebuild K0
        K0_ms = self.Ks[0]

        K0 = t.buildBlock(z16, K0_ms[0], K0_ms[1], z16)

        K = (K4 + K3 + K2 + K1 + K0)
        #K = (K3 + K2 + K1 + K0) - 3 * i16

        #print(K3.round(1))
        #print(K2.round(1))
        '''print(K1.round(1))
        print(K0.round(1))'''

        import sys
        np.set_printoptions(threshold=sys.maxsize)

        #print((K - covMat).round(1))
        #print(K.round(1))
        #print(covMat.round(1))

        #plt.matshow(K)
        #plt.show()

        b = np.arange(len(K0)).T

        K4i = inv(K4)
        b = K4i @ b

        K3i = self.sherWood(K4i, K3, i32)
        b = K3i @ b

        K2i = self.sherWood(K3i, K4i@K2, i32)
        b = K2i @ b

        K1i = self.sherWood(K2i, K3i@K4i@K1, i32)
        b = K1i @ b
                            # U, Vt, I
        K0i = self.sherWood(K1i, K2i@K3i@K4i@K0, i32)
        b = K0i @ b

        #B1i = (i16 + inv( i16 + K3i@K2i ) @ K3i@K1)
        #B0 = inv( i16 + K3i@K2i ) @ K3i@K0

        #C1i = i16 + K2i @ K3i @ K1
        #C0 = K2i @ K3i @ K0

        #finalB = self.sherWood(C1i, C0, i32) @ b1

        #print(finalB.round(2))
        #print(b.round(2))

        return b
