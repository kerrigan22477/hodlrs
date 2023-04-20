import numpy as np
from buildHodlr import buildHodlr
from kmeans import Kmeans
from solveForX import SolveForX
from multiply import Multiply
from invert import Invert
from reorderPoints import ReorderPoints
from numpy.linalg import inv
from sklearn.metrics import pairwise_distances

class Main:
    def __init__(self):
        pass

    def generate_points(self, k, size):
        s = size // k
        loc = [x for x in range(0, (5 * k + 1), 5)]
        xs = []
        ys = []
        # s = num_datapoints // 2 because this is the size of each cluster
        for i in range(k):
            x = np.random.normal(loc=loc[i], size=s)
            y = np.random.normal(loc=loc[i], size=s)

            xs = np.concatenate((xs, x), axis=0)
            ys = np.concatenate((ys, y), axis=0)

        return np.array([list(pair) for pair in zip(xs, ys)])


    def main(self, r, k, l, approx, data):

        # kmeans
        km = Kmeans()
        clusters, centroids = km.kmeans(data, k, 1000)

        # organize points by clusters
        re = ReorderPoints()
        new_points = re._reorderPoints(data, clusters, k)

        # compute covariance matrix
        covMat = np.array(pairwise_distances(new_points, new_points))
        covMat = np.exp(-(covMat**2)/(l**2)) + np.eye(covMat.shape[0]) * .01

        # build hodlr
        b = buildHodlr(k, r)
        hodlr, root = b.buildHodlr(covMat, approx)

        # solveForX
        s = SolveForX()
        b = np.arange(len(covMat)).T
        max_level = np.log2(len(covMat)) - 1
        y, Kli, update, next_update = s.solveForX(b, root, int(max_level))

        # compare to numpy tested version
        test = np.linalg.solve(covMat, b)

        print(y.round(2))
        print(test.round(2))
        print((test - y).round(2))

        '''np.set_printoptions(suppress=True)
        i = Invert()
        Ai = i.invert(root)
        print(Ai.round(2))
        # print(Ai)
        B = inv(covMat).round(2)
        # B = inv(covMat)
        print('---')
        print((Ai - B).round(2))'''

        '''m = Multiply()
        q = np.arange(len(covMat)).T
        y = m.multiply(root, q)

        test = covMat @ q
        print(y.round(2))
        print(test.round(2))
        print((test - y).round(2))'''

        return hodlr, root, covMat

if __name__ == "__main__":
    m = Main()
    size = 128
    r = size // 2
    k = 2
    l = 1
    approx = True
    data_points = m.generate_points(k, size)
    m.main(r, k, l, approx, data_points)