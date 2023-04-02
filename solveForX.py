import numpy as np
from numpy.linalg import inv
from tools import Tools
from BST import Node, Leaf
import matplotlib.pyplot as plt

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

    def factor(self, root, level):

        if root.left is not None:
            self.factor(root.left, level + 1)
            self.factor(root.right, level + 1)

            A = root.value
            U1, S1, VT1, U2, S2, VT2 = A[0], A[1], A[2], A[3], A[4], A[5]

            uR = (U1.dot(S1)).dot(VT1)
            lL= (U2.dot(S2)).dot(VT2)
            #upperRight = [U1, S1, VT1]
            #lowerLeft = [U2, S2, VT2]

            try:
                #self.Ks[level] = self.Ks[level] + [upperRight, lowerLeft]
                self.Ks[level] = self.Ks[level] + [uR, lL]
            except KeyError:
                #self.Ks[level] = [upperRight, lowerLeft]
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

    def solveForX_bruteForce(self, n, b, root):
        t = Tools()
        self.factor(root, 0)


        levels = self.Ks

        '''
        for i in range(len(levels)-1):
            print(i)
            print(len(levels[i]))
            for x in range(len(levels[i])):
                U1 = self.Ks[i][x][0]
                S1 = self.Ks[i][x][1]
                VT1 = self.Ks[i][x][2]
                first = U1.dot(S1).dot(VT1)
                print(first.round(2))
                print('!!!!!!!!!!!!!!!!!!!!!')

        print('----------')
        for x in range(len(levels[3])):
            print(levels[3][x])'''


        z2 = np.zeros((2, 2))
        z4 = np.zeros((4, 4))
        z8 = np.zeros((8, 8))
        i2 = np.identity(2)
        i4 = np.identity(4)
        i8 = np.identity(8)
        i16 = np.identity(16)

        # rebuild K3
        a = t.buildBlock(levels[3][0], z2, z2, levels[3][1])
        b = t.buildBlock(levels[3][2], z2, z2, levels[3][3])
        c = t.buildBlock(levels[3][4], z2, z2, levels[3][5])
        d = t.buildBlock(levels[3][6], z2, z2, levels[3][7])

        K3_UL = t.buildBlock(a, z4, z4, b)
        K3_LR = t.buildBlock(c, z4, z4, d)

        K3 = t.buildBlock(K3_UL, z8, z8, K3_LR)

        # rebuild K2
        #K2_ms = self.undoSVD(2)
        K2_ms = self.Ks[2]

        # should be i's
        use = z2
        a = t.buildBlock(use, K2_ms[0], K2_ms[1], use)
        b = t.buildBlock(use, K2_ms[2], K2_ms[3], use)
        c = t.buildBlock(use, K2_ms[4], K2_ms[5], use)
        d = t.buildBlock(use, K2_ms[6], K2_ms[7], use)

        K2_UL = t.buildBlock(a, z4, z4, b)
        K2_LR = t.buildBlock(c, z4, z4, d)

        K2 = t.buildBlock(K2_UL, z8, z8, K2_LR)

        #rebuild K1
        #K1_ms = self.undoSVD(1)
        K1_ms = self.Ks[1]

        # should be i's
        a = t.buildBlock(z4, K1_ms[0], K1_ms[1], z4)
        d = t.buildBlock(z4, K1_ms[2], K1_ms[3], z4)

        K1 = t.buildBlock(a, z8, z8, d)

        #rebuild K0
        '''K0_ms = []

        U1 = self.Ks[0][0][0]
        S1 = self.Ks[0][0][1]
        VT1 = self.Ks[0][0][2]
        K0_ms.append(U1.dot(S1).dot(VT1))

        U1 = self.Ks[0][1][0]
        S1 = self.Ks[0][1][1]
        VT1 = self.Ks[0][1][2]
        K0_ms.append(U1.dot(S1).dot(VT1))'''

        K0_ms = self.Ks[0]

        # should be i's
        K0 = t.buildBlock(z8, K0_ms[0], K0_ms[1], z8)

        #plt.matshow(K0)
        #plt.show()


        #print(K3.round(1))
        #print(K2.round(1))
        #print(K1.round(1))
        #print(K0.round(1))

        K = (K3+K2+K1+K0)
        #print(K.round(1))

        #plt.matshow(K)
        #plt.show()

        # Do maths
        x = np.array([-579.68564799,  520.63842807, -190.82583598,  170.68069913, -161.69745997,
   65.3555604 ,  -72.7476185 ,  261.73045681 ,  33.85947749,  -77.48355648,
   60.24764757, -124.92497873 ,  71.65899951 , -20.65095809 , 110.14746495,
  -30.19240434])
        b = np.arange(len(K0))
        K3i = inv(K3)
        '''
        K2i = self.sherWood(K3i, K2, i16)
        K1i = self.sherWood(K2i, K3i @ K1, i16)
        K0i = self.sherWood(K1i, K3i @ K2i @ K0, i16)'''
        K2i = inv(K2)
        K1i = inv(K1)
        K0i = inv(K0)

        LHS = (K3+K2+K1+K0)@x
        RHS = b

        #print('X: ' + str(x.round(2)))
        print(LHS.round(2))
        print(RHS.round(2))
       # print(b)

        b0 = K

        '''
        K3i = inv(K3)
        b3 = K3i @ b

        K2i = self.sherWood(K3i, K2, i16)
        b2 = K2i @ b3

        K1i = self.sherWood(K2i, K3i@K1, i16)
        b1 = K1i @ b2

        K0i = self.sherWood(K1i, K3i@K2i@K0, i16)

        b0 = K0i @ b1
        '''

        # invert and solve
        #K2i = inv(K2)
        #b2 = K2i @ b

        # invert and solve
        # U, Vt, b, I
        #K1i = self.sherWood(K2i, K1, i8)
        #b1 = K1i @ b2

        # invert and solve
        # multiply by inverse of K2 as if we had factored K2 out
        # K0 needs to have K2 factored out before we factor out K1, hence Vt = K2i@K0
        # U, Vt, I
        #K0i = self.sherWood(K1i, K2i @ K0, i8)


        #b0 = K0i @ b1
        return b0

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

            # U, Vt, b, I
            # rebuild update needed for current matrix w/ previous inverse matrices
            update = t.buildBlock(update_UL, z, z, update_LR)
            next_update = t.buildBlock(next_update_UL, z, z, next_update_LR)

            K1i = self.sherWood(prev_K, (update@K1), I)

            # update b w/ previous inverse matrices
            self.x = np.concatenate((x1, x2), axis=0)
            self.x = K1i@self.x

            return self.x, K1i, next_update, next_update@K1i


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
        i2 = np.identity(2)
        i4 = np.identity(4)
        i8 = np.identity(8)
        i16 = np.identity(16)

        # rebuild K3
        a = t.buildBlock(levels[3][0], z2, z2, levels[3][1])
        b = t.buildBlock(levels[3][2], z2, z2, levels[3][3])
        c = t.buildBlock(levels[3][4], z2, z2, levels[3][5])
        d = t.buildBlock(levels[3][6], z2, z2, levels[3][7])

        K3_UL = t.buildBlock(a, z4, z4, b)
        K3_LR = t.buildBlock(c, z4, z4, d)

        K3 = t.buildBlock(K3_UL, z8, z8, K3_LR)

        # rebuild K2
        #K2_ms = self.undoSVD(2)
        K2_ms = self.Ks[2]

        a = t.buildBlock(z2, K2_ms[0], K2_ms[1], z2)
        b = t.buildBlock(z2, K2_ms[2], K2_ms[3], z2)
        c = t.buildBlock(z2, K2_ms[4], K2_ms[5], z2)
        d = t.buildBlock(z2, K2_ms[6], K2_ms[7], z2)
        '''a = t.buildBlock(i2, K2_ms[0], K2_ms[1], i2)
        b = t.buildBlock(i2, K2_ms[2], K2_ms[3], i2)
        c = t.buildBlock(i2, K2_ms[4], K2_ms[5], i2)
        d = t.buildBlock(i2, K2_ms[6], K2_ms[7], i2)'''

        K2_UL = t.buildBlock(a, z4, z4, b)
        K2_LR = t.buildBlock(c, z4, z4, d)

        K2 = t.buildBlock(K2_UL, z8, z8, K2_LR)

        # rebuild K1
        #K1_ms = self.undoSVD(1)
        K1_ms = self.Ks[1]


        '''a = t.buildBlock(i4, K1_ms[0], K1_ms[1], i4)
        d = t.buildBlock(i4, K1_ms[2], K1_ms[3], i4)'''
        a = t.buildBlock(z4, K1_ms[0], K1_ms[1], z4)
        d = t.buildBlock(z4, K1_ms[2], K1_ms[3], z4)

        K1 = t.buildBlock(a, z8, z8, d)

        # rebuild K0
        K0_ms = self.Ks[0]

        '''K0 = t.buildBlock(i8, K0_ms[0], K0_ms[1], i8)'''
        K0 = t.buildBlock(z8, K0_ms[0], K0_ms[1], z8)

        K = (K3 + K2 + K1 + K0)
        #K = (K3 + K2 + K1 + K0) - 3 * i16

        '''print(K3.round(1))
        print(K2.round(1))
        print(K1.round(1))
        print(K0.round(1))'''

        print((K - covMat).round(1))
        print(K.round(1))
        print(covMat.round(1))

        #plt.matshow(K)
        #plt.show()

        # Do maths
        '''x = np.array([-372.00641187, 224.77515493, -77.15161439, 227.87207448, -309.25801318,
                      329.96835622, 143.92113187, -150.76826154, 142.3190304, -200.70929676,
                      51.4164312, 60.72415455, -109.50084279, -110.46445274, -5.28376826,
                      189.35892997])

        b = np.arange(len(K0))'''
        print(x.round(2))
        #LHS = (K3 + K2 + K1 + K0) @ x
        LHS = K @ x
        b = np.arange(len(K0))
        # print('X: ' + str(x.round(2)))
        print(LHS.round(2))
        #print(b.round(2))

        #print((covMat @ x).round(2))

        K3i = inv(K3)
        b3 = K3i @ b
        K2i = self.sherWood(K3i, K2, i16)
        b2 = K2i @ b3
        K1i = self.sherWood(K2i, K3i@K1, i16)
        b1 = K1i @ b2
        K0i = self.sherWood(K1i, K3i@K2i@K0, i16)
        b0 = K0i @ b1

        print(b0.round(2))

        return b0


