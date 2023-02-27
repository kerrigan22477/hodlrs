import numpy as np

class Split:
    def __init__(self):
        self.A11 = []
        self.A12 = []
        self.A21 = []
        self.A22 = []

    def split(self, A):
        #split A vertically
        list = np.split(A, 2)
        #split a horizontally
        tmp1 = np.hsplit(list[0], 2)
        tmp2 = np.hsplit(list[1], 2)
        self.A11 = tmp1[0]
        self.A12 = tmp1[1]
        self.A21 = tmp2[0]
        self.A22 = tmp2[1]

        return self.A11, self.A12, self.A21, self.A22