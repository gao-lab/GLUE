"""
Author: Kai Cao
Utils for Pamona
"""
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import random
from itertools import chain
from matplotlib import pyplot as plt


def Pamona_geodesic_distances(X, num_neighbors, mode="distance", metric="minkowski"):

    assert (mode in ["connectivity", "distance"]), "Norm argument has to be either one of 'connectivity', or 'distance'. "
    if mode=="connectivity":
        include_self=True
    else:
        include_self=False
    knn = kneighbors_graph(X, num_neighbors, n_jobs=-1, mode=mode, metric=metric, include_self=include_self)
    connected_components = sp.csgraph.connected_components(knn, directed=False)[0]
    dist = sp.csgraph.dijkstra(knn, directed=False)
    connected_element = []

    ## for local connectively
    if connected_components != 1:
        inf_matrix = []

        for i in range(len(X)):
            inf_matrix.append(list(chain.from_iterable(np.argwhere(np.isinf(dist[i])))))

        for i in range(len(X)):
            if i==0:
                connected_element.append([0])
            else:
                for j in range(len(connected_element)+1):
                    if j == len(connected_element):
                        connected_element.append([])
                        connected_element[j].append(i)
                        break
                    if inf_matrix[i] == inf_matrix[connected_element[j][0]]:
                        connected_element[j].append(i)
                        break

        components_dist = []
        x_index = []
        y_index = []
        components_dist.append(np.inf)
        x_index.append(-1)
        y_index.append(-1)
        for i in range(connected_components):
            for j in range(i):
                for num1 in connected_element[i]:
                    for num2 in connected_element[j]:
                        if np.linalg.norm(X[num1]-X[num2])<components_dist[len(components_dist)-1]:
                            components_dist[len(components_dist)-1]=np.linalg.norm(X[num1]-X[num2])
                            x_index[len(x_index)-1] = num1
                            y_index[len(y_index)-1] = num2
                components_dist.append(np.inf)
                x_index.append(-1)
                y_index.append(-1)

        components_dist = components_dist[:-1]
        x_index = x_index[:-1]
        y_index = y_index[:-1]

        sort_index = np.argsort(components_dist)
        components_dist = np.array(components_dist)[sort_index]
        x_index = np.array(x_index)[sort_index]
        y_index = np.array(y_index)[sort_index]

        for i in range(len(x_index)):
            knn = knn.todense()
            knn = np.array(knn)
            knn[x_index[i]][y_index[i]] = components_dist[i]
            knn[y_index[i]][x_index[i]] = components_dist[i]
            knn = sp.csr_matrix(knn)
            connected_components = sp.csgraph.connected_components(knn, directed=False)[0]
            dist = sp.csgraph.dijkstra(knn, directed=False)
            if connected_components == 1:
                break

    return dist/dist.max()

def get_spatial_distance_matrix(data, metric="euclidean"):
    Cdata= sp.spatial.distance.cdist(data,data,metric=metric)
    return Cdata/Cdata.max()

def unit_normalize(data, norm="l2", bySample=True):
    """
    Default norm used is l2-norm. Other options: "l1", and "max"
    If bySample==True, then we independently normalize each sample. If bySample==False, then we independently normalize each feature
    """
    assert (norm in ["l1","l2","max"]), "Norm argument has to be either one of 'max', 'l1', or 'l2'."

    if bySample==True:
        axis=1
    else:
        axis=0

    return normalize(data, norm=norm, axis=axis)


def zscore_standardize(data):
    scaler=StandardScaler()
    scaledData=scaler.fit_transform(data)
    return scaledData


def init_random_seed(manual_seed):
    seed = None
    if manual_seed is None:
        seed = random.randint(1,10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)


def Interval_Estimation(Gc, interval_num=20):
    Gc = np.array(Gc)
    Gc_last_col = Gc[0:-1,-1]
    Gc_max = np.max(Gc_last_col)
    Gc_min = np.min(Gc_last_col)
    Gc_interval = Gc_max - Gc_min

    row = np.shape(Gc)[0]-1
    col = np.shape(Gc)[1]-1
    count = np.zeros(interval_num)

    interval_value = []
    for i in range(interval_num+1):
        interval_value.append(Gc_min+(1/interval_num)*i*Gc_interval)

    for i in range(row):
        for j in range(interval_num):
            if Gc[i][col] >= interval_value[j] and Gc[i][col] < interval_value[j+1]:
                count[j] += 1
            if Gc[i][col] == interval_value[j+1]:
                count[interval_num-1] += 1

    print('count', count)

    fig = plt.figure(figsize=(10, 6.5))

    a = list(range(interval_num))
    a = list(map(str,a))
    font_label = {
             'weight': 'normal',
             'size': 25,
         }

    plt.plot(a,count,'k')

    for i in range(interval_num):
        plt.plot(a[i], count[i], 's', color='k')

    plt.xticks(fontproperties = 'Arial', size = 18)
    plt.yticks(fontproperties = 'Arial', size = 18)
    plt.xlabel('interval', font_label)
    plt.ylabel('number', font_label)

    plt.show()
