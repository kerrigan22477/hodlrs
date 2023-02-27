import numpy as np
from buildHodlr import buildHodlr
from solveForX import SolveForX
from tools import Tools
from numpy.linalg import inv

def sherWood(U, Vt, I):
    #i8 + K1i @ (K2i @ K0)
    #x = b - (U @ inv(I + Vt@U) @ Vt @ b)
    inverse = I - (U @ inv(I + Vt@U) @ Vt)
    # K0 after factoring
    return inverse

covMat = np.array([[1., 0.64, 0.57, 0.89, 0.41, 0.64, 0.01, 0.02],
                   [0.64, 1., 0.89, 0.57, 0.64, 0.41, 0.11, 0.15],
                   [0.57, 0.89, 1., 0.64, 0.89, 0.57, 0.15, 0.17],
                   [0.89, 0.57, 0.64, 1., 0.57, 0.89, 0.02, 0.02],
                   [0.41, 0.64, 0.89, 0.57, 1., 0.64, 0.17, 0.15],
                   [0.64, 0.41, 0.57, 0.89, 0.64, 1., 0.02, 0.02],
                   [0.01, 0.11, 0.15, 0.02, 0.17, 0.02, 1., 0.89],
                   [0.02, 0.15, 0.17, 0.02, 0.15, 0.02, 0.89, 1.],
                   ])

b = buildHodlr()
s = SolveForX()
t = Tools()
data = np.array([[3.0, 8.0], [7.0, 8.0], [9.0, 10.0], [7.0, 10.0], [3.0, 9.0], [9.0, 8.0], [7.0, 9.0], [9.0, 9.0]])
hodlr, root, points = b.buildHodlr(data, 0, 16, covMat, False)
#print(hodlr.printTree(root))

levels = s.factor(root, 0)


'''
for l in levels[0]:
    U1 = l[0]
    S1 = l[1]
    VT1 = l[2]
    uppR = (U1.dot(S1)).dot(VT1)
    print(l)
    print(uppR)
    print('----')
'''


#K2 = K1 = K0 = np.zeros((8,8))

z2 = np.zeros((2,2))
z4 = np.zeros((4,4))
i2 = np.identity(2)
i8 = np.identity(8)

a = t.buildBlock(levels[2][0], z2, z2, levels[2][1])
d = t.buildBlock(levels[2][2], z2, z2, levels[2][3])

K2 = t.buildBlock(a, z4, z4, d)

b = np.arange(len(covMat)).T

K2i = inv(K2)
b2 = K2i@b

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

# build the original K
K1 = t.buildBlock(a, z4, z4, d)

#print(K2i.round(2))

# multiply by inverse of K2 as if we had factored K2 out
K1_after_factoring = i8 + K2i@K1
#K1i = inv(K1_after_factoring)
#b1 = K1i@b2

#print(K1i.round(2))


'''' SHERMAN EXPERIEMENT '''

# U, Vt, b, I
K1i = sherWood(K2i, K1, i8)
#print((test_K1i - K1i).round(2))
b1 = K1i@b2

''''SHERMAN EXPERIEMENT '''


#print(K1_after_factoring.round(2))
#print(b1.round(2))


U1 = levels[0][0][0]
S1 = levels[0][0][1]
VT1 = levels[0][0][2]
uppR = (U1.dot(S1)).dot(VT1)

U2 = levels[0][1][0]
S2 = levels[0][1][1]
VT2 = levels[0][1][2]
lowL = (U2.dot(S2)).dot(VT2)

K0 = t.buildBlock(z4, uppR, lowL, z4)

# multiply by inverse of K2 as if we had factored K2 out
K0_after_factoring = i8 + K1i@(K2i@K0)
#K0i = inv(K0_after_factoring)

'''' SHERMAN EXPERIEMENT '''

# U, Vt, b, I
#K1_after_factoring = i8 + K2i@K1
K0i = sherWood(K1i, K2i@K0, i8)
#print((test_K0i - K0i).round(2))

''''SHERMAN EXPERIEMENT '''

b0 = K0i@b1


x = np.array([-26.79183204,  25.88875213, -48.04235809,  39.31662254,  26.06864796, -12.89726922,  -4.34186584 , 11.24530103])

#print(rebuilt.round(2))

test = np.linalg.solve(covMat, b)
#print(test)
print((test-b0).round(2))

