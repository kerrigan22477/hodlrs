import numpy as np
from tools import Tools

class Multiply:
    def __init__(self):
        self.u = []

    def multiply(self, root, q):
        t = Tools()
        A = root.value

        A12, A21, A11, A22 = t.splitA(A, root)
        q1, q2 = t.splitQ(q)

        # if A11 is full matrix (has no babies)
        if root.left.left is None:

            u1 = A11.dot(q1) + A12.dot(q2)
            u2 = A22.dot(q2) + A21.dot(q1)
            self.u = np.concatenate((u1, u2), axis=0)
            return self.u

        # if A11 is a SVD and therefore has babies
        else:
            upperLeft = self.multiply(root.left, q1)
            lowerRight = self.multiply(root.right, q2)

            u1 = upperLeft + A12.dot(q2)
            u2 = lowerRight + A21.dot(q1)
            self.u = np.concatenate((u1, u2), axis=0)

            return self.u