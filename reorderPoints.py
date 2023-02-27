import numpy as np
from kmeans import Kmeans

class ReorderPoints:
    def __init__(self):
        self.points = []
        self.finalPoints = []

    def splitPoints(self, points, clusters):
        # find indexes where for each cluster
        c1 = np.where(clusters == 0)
        c2 = np.where(clusters == 1)
        # pull points using the indices
        c1_points = points[c1]
        c2_points = points[c2]

        return c1_points, c2_points

    def smallerKmeans(self, points, k):
        # kmeans
        km = Kmeans()
        clusters, centroids = km.kmeans(points, k, 1000)

        # compute c1 c2
        c1_points, c2_points = self.splitPoints(points, clusters)
        return c1_points, c2_points

    def _reorderPoints(self, points, clusters, k):
        self.points = points

        c1_points, c2_points = self.splitPoints(self.points, clusters)

        self.reorderPoints(c1_points, k)
        self.reorderPoints(c2_points, k)

        return self.finalPoints

    def reorderPoints(self, points, k):

        if len(points) > 4:
            c1, c2 = self.smallerKmeans(points, k)
        else:
            self.finalPoints.extend(points.tolist())
            c1 = c2 = []

        if len(points) <= 4:
            return c1, c2

        else:
            self.reorderPoints(c1, k)
            self.reorderPoints(c2, k)
