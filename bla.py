import numpy as np

bico= np.loadtxt("bico_50.txt", dtype=float)
kmeans=np.loadtxt("kmeans_res.txt", dtype=float)
avg_bico=[sum(bico[range(i)])/(i+1.) for i in range(len(kmeans))]
avg_kmeans=[sum(kmeans[range(i)])/(i+1.) for i in range(len(kmeans))]

import matplotlib.pyplot as plt
plt.plot(avg_bico,"r-",label="BICO")
plt.plot(avg_kmeans,"g-",label="Kmeans")
plt.xlabel("Batch Number")
plt.ylabel("Cluster Cost")
plt.legend()
plt.savefig("cluster compare.png")
plt.show()