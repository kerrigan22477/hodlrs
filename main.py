import numpy as np
from buildHodlr import buildHodlr
from kmeans import Kmeans
from solveForX import SolveForX
from reorderPoints import ReorderPoints
from numpy.linalg import inv
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

class Main:
    def __init__(self):
        self.data = []
        self.k = 0
        self.r = 0
        self.index = {}

    def main(self):

        s = 16
        x1 = np.random.normal(loc=3.0, size=s)
        y1 = np.random.normal(loc=2.0, size=s)
        x2 = np.random.normal(loc=9.0, size=s)
        y2 = np.random.normal(loc=7.0, size=s)

        xs = np.concatenate((x1, x2), axis=0)
        ys = np.concatenate((y1, y2), axis=0)

        self.data = np.array([list(pair) for pair in zip(xs, ys)])

        '''plt.scatter(*self.data[:].T)
        plt.show()'''

        # kmeans
        km = Kmeans()
        self.k = 2
        self.clusters, self.centroids = km.kmeans(self.data, self.k, 1000)


        r = ReorderPoints()
        new_points = r._reorderPoints(self.data, self.clusters, self.k)

        # compute covariance matrix
        covMat = np.array(pairwise_distances(new_points, new_points))
        l = 3
        covMat = np.exp(-(covMat**2)/(l**2))

        # build hodlr
        approx = False
        self.r = 8
        b = buildHodlr()
        hodlr, root, points = b.buildHodlr(self.k, self.r, covMat, approx)
        #hodlr.printTree(root)

        # solve For X
        np.set_printoptions(suppress=True)
        s = SolveForX()
        b = np.arange(len(covMat)).T
        test = np.linalg.solve(covMat, b)
        #print(test.round(2))
        max_level = np.log2(len(covMat)) - 1

        y, Kli, update, next_update = s.solveForX(b, root, int(max_level))

        print(y.round(2))
        print(test.round(2))
        print((test - y).round(2))

if __name__ == "__main__":
    m = Main()
    m.main()