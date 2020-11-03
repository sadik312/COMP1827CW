import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans

x = [1, 5, 1.5, 8, 1, 9]
y = [1, 8, 1.8, 8, 0.6, 11]

plt.scatter(x,y)
plt.show()

# numpy array of data
#X = np.array([[1, 2], [5, 8], [1.5, 1.8], [8, 8], [1, 0.6], [9, 11]])
X = np.array(list(zip(x,y))) # easier, better for larger datasets

# flat clustering where we tell ML algo how many clusters we want
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

# centre marker of cluster
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print(centroids)
print(labels)

colors = ["g.", "r.", "c.", "b."]

for i in range(len(X)): 
    print("Coordinate:", X[i], "label:", labels[i])
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)

plt.scatter(centroids[:, 0], centroids[:, 1], marker = "x", s=150, linewidths = 5, zorder = 10)

plt.show()