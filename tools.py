import numpy as np

class Tools:

    def printDim(self, m):
        if m.ndim == 1:
            rows = 1
            cols = len(m)
        else:
            rows = len(m)
            cols = np.size(m, 1)
        return str(rows) + 'x' + str(cols)

    def splitQ(self, q):
        splits = np.hsplit(q, 2)
        q1 = np.array(splits[0])
        q2 = np.array(splits[1])
        return q1, q2

    def splitA(self, A, root):
        U1, S1, VT1, U2, S2, VT2 = A[0], A[1], A[2], A[3], A[4], A[5]
        A12 = (U1.dot(S1)).dot(VT1)
        A21 = (U2.dot(S2)).dot(VT2)
        A11 = root.left.value
        A22 = root.right.value

        return A12, A21, A11, A22

    def splitAi(self, A):
        # split A vertically
        list = np.split(A, 2)
        # split a horizontally
        tmp1 = np.hsplit(list[0], 2)
        tmp2 = np.hsplit(list[1], 2)
        A11 = tmp1[0]
        A12 = tmp1[1]
        A21 = tmp2[0]
        A22 = tmp2[1]

        return A12, A21, A11, A22

    def buildBlock(self, a, b, c, d):
        block = np.block([
            [a, b],
            [c, d]
        ])
        return block