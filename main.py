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

        '''self.data = np.array(
            [[1.0, 0.0], [1.0, 2.0], [1.0, 4.0], [1.0, 6.0], [1.0, 8.0], [1.0, 10.0], [1.0, 12.0], [1.0, 14.0],
             [3.0, 0.0], [3.0, 2.0], [3.0, 4.0], [3.0, 6.0], [3.0, 8.0], [3.0, 10.0], [3.0, 12.0], [3.0, 14.0],
             [7.0, 0.0], [7.0, 2.0], [7.0, 4.0], [7.0, 6.0], [7.0, 8.0], [7.0, 10.0], [7.0, 12.0], [7.0, 14.0],
             [9.0, 0.0], [9.0, 2.0], [9.0, 4.0], [9.0, 6.0], [9.0, 8.0], [9.0, 10.0], [9.0, 12.0], [9.0, 14.0]])'''
        self.data = np.array(
           [[3.0, 8.0], [7.0, 8.0], [9.0, 10.0], [7.0, 10.0], [3.0, 9.0],
            [9.0, 8.0], [7.0, 9.0], [9.0, 9.0], [0.0, 9.0], [0.0, 8.0],
            [3.0, 7.0], [0.0, 7.0], [0.0, 9.5], [3.0, 9.5], [8.0, 8.0], [8.0, 8.5]])

        self.data = np.array(
            [[3.0, 8.0], [7.0, 8.0], [9.0, 10.0], [7.0, 10.0], [3.0, 9.0], [9.0, 8.0], [7.0, 9.0], [9.0, 9.0]])

        s = 8
        x1 = np.random.normal(loc=3.0, size=s)
        y1 = np.random.normal(loc=2.0, size=s)
        x2 = np.random.normal(loc=9.0, size=s)
        y2 = np.random.normal(loc=7.0, size=s)

        xs = np.concatenate((x1, x2), axis=0)
        ys = np.concatenate((y1, y2), axis=0)

        self.data = np.array([list(pair) for pair in zip(xs, ys)])

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

        from testing import Test
        t = Test()
        covMat = t.returnmatrix()

        # build hodlr
        approx = False
        self.r = 16
        b = buildHodlr()
        hodlr, root, points = b.buildHodlr(self.k, self.r, covMat, approx)
        #hodlr.printTree(root)

        np.set_printoptions(suppress=True)
        s = SolveForX()
        b = np.arange(len(covMat))
        test = np.linalg.solve(covMat, b)
        #print(test.round(2))
        max_level = np.log2(len(covMat)) - 1
        #x = s.solveForX_bruteForce(len(covMat), b, root)
        y, Kli, update, next_update = s.solveForX(b, root, int(max_level))

        ex = s.solveForX16(len(covMat), b, root, test, covMat)

        #print(test.round(2))
        #print(ex.round(2))
        #print((test - ex).round(2))

        #plt.matshow(covMat)
        #plt.show()


        '''colors = ['C' + str(i) for i in range(self.k)]
        for i in range(self.k):
            # plot points for curr cluster
            plt.scatter(self.data[self.clusters == i, 0], self.data[self.clusters == i, 1], label='cluster ' + str(i),
                        color=colors[i])
            # plot centroid for curr cluster
            plt.scatter(self.centroids[i, 0], self.centroids[i, 1],
                        marker='o',
                        s=100,  # size
                        linewidths=2,
                        color=colors[i],
                        edgecolors='black')

        plt.show()'''



if __name__ == "__main__":
    m = Main()
    m.main()