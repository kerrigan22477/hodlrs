import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from celluloid import Camera
import time

class Kmeans:

    def __init__(self):
        self.clusters = []
        self.data = []
        self.k = 0

    def update_centroids(self, clusters):
        # new centroid = mean of all its cluster's points
        # if a cluster has no data, replace it with a random data point
        return np.array([np.mean(self.data[clusters == i], axis=0)
                         if np.sum(clusters == i) != 0
                        # else self.data[np.random.randint(m)]
                         else self.data[np.random.randint(len(self.data)-1)]
                         for i in range(self.k)])


    def update_clusters(self, centroids):
        # check every point and assign it to the cluster with the nearest centroid
        dist = pairwise_distances(self.data, centroids)
        return np.argmin(dist, axis=1)


    def kmeans(self, data, k, max_iterations=1000):
        # initialize iteration counter
        it = 0
        done = False

        # number of points
        self.data = data
        self.k = k
        num_datapoints = self.data.shape[0]

        # initialization
        # choose random data points from set to be initial centroids
        centroids = self.data[np.random.choice(num_datapoints, self.k, replace=False)]
        # find distances between each point and centroid
        dist = pairwise_distances(self.data, centroids)
        # return min distance values for each point
        # (aka which cluster is closest to each point)
        clusters = np.argmin(dist, axis=1)

        #set up camera
        #camera = Camera(plt.figure())
        #colors = ['C' + str(i) for i in range(self.k)]

        while not done and it < max_iterations:

            # update centroids
            centroids = self.update_centroids(clusters)

            # check every point and assign it to the cluster with the nearest centroid
            new_clusters = self.update_clusters(centroids)

            # if new clusters = old clusters we're done
            if np.sum(clusters != new_clusters) == 0:
                done = True
            clusters = new_clusters

            it += 1  # increment iteration counter by 1

            # display current graph
            # iterate through clusters

            '''
            for i in range(self.k):
                # plot points for curr cluster
                plt.scatter(self.data[clusters == i, 0], self.data[clusters == i, 1], label='cluster ' + str(i),
                            color=colors[i])
                # plot centroid for curr cluster
                plt.scatter(centroids[i, 0], centroids[i, 1],
                            marker='o',
                            s=100,  # size
                            linewidths=2,
                            color=colors[i],
                            edgecolors='black')


            camera.snap()
            time.sleep(0.1)
            '''



        # display animation
        #anim = camera.animate()
       # plt.show()


        return clusters, centroids