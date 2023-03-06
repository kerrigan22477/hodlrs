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
        b = buildHodlr()


        self.data = np.array(
            [[1.0, 8.0], [1.0, 10.0], [1.0, 9.0], [1.0, 11.0],
             [3.0, 8.0], [3.0, 10.0], [3.0, 9.0], [3.0, 11.0],
             [7.0, 8.0], [7.0, 9.0], [7.0, 10.0], [7.0, 11.0],
             [9.0, 8.0], [9.0, 9.0], [9.0, 10.0], [9.0, 11.0]])


        # try np.random.seed for some consistency
        self.data = 10*np.random.rand(16,2)
        #print(self.data)

        #self.data = np.array([[3.0, 8.0], [7.0, 8.0], [9.0, 10.0], [7.0, 10.0], [3.0, 9.0], [9.0, 8.0], [7.0, 9.0], [9.0, 9.0]])

        #plt.scatter(*self.data[:].T)
        #plt.show()

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

        print(covMat)

        # display covMat
        #plt.matshow(covMat)
        #plt.show()

        from testing import Test
        t = Test()
        #covMat = t.returnmatrix()

        # build hodlr
        approx = False
        hodlr, root, points = b.buildHodlr(self.data, self.k, 16, covMat, approx)
        #print(hodlr.printTree(root))

        #plt.matshow(covMat)
       # plt.show()

        np.set_printoptions(suppress=True)
        s = SolveForX()
        b = np.arange(len(covMat))
        max_level = np.log2(len(covMat)) - 1
        #x = s.solveForX_bruteForce(len(covMat), b, root)
        y, Kli, update, next_update = s.solveForX(b, root, int(max_level))
        test = np.linalg.solve(covMat, b)
        print(test.round(2))
        mytest = covMat@test
        #print(mytest)
        print(y.round(2))
        print((test - y).round(2))
        #print('-----------')

        '''
        max_level = np.log2(len(covMat)) - 1
        y, Kli, update, next_update = s.solveForX(b, root, int(max_level))

        test = np.linalg.solve(covMat, b)
        print(test.round(2))
        print(y.round(2))
        print((test-y).round(2))
        '''


if __name__ == "__main__":
    m = Main()
    m.main()