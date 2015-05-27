from numpy import *
import matplotlib.pyplot as plt

#----------------------------------------------------------------------
def isodata(X, c, theta_n, theta_s, theta_d, L, I, init_z, ip):
    X = mat(X)
    init_z = mat(init_z)
    n_samples = X.shape[0]
    z = {}  #the center points dictionary
    for i in range(init_z.shape[0]):
        z[i] = init_z[i, :]
    labels = {}  #the labels dictionary
    while True:
        print('ip = %d' % ip)
        labels = clusterByDistance(z, X)
        if ip > I:
            break
        #remove the clusters of which the sample numbers are lower than theta_n
        for i in z.keys():
            if len(labels[i]) < theta_n:
                del z[i]
                print('One cluster has been canceled because of too few samples')
                continue
        for i in z.keys():
            z[i] = sum(X[labels[i], :], 0) / float(len(labels[i]))        
        if ip == I:
            theta_d = 0
        else:
            Nc = len(z.keys())
            if Nc <= c / 2.0 or (Nc < 2 * c and mod(ip, 2) == 1):            
                z, splited = cluster_split(z, labels, X, c, theta_s, theta_n)
                if splited:
                    ip += 1
                    continue
        z = cluster_merge(z, labels, L, theta_d)
        ip += 1
    return labels

def cluster_merge(z, labels, L, theta_d):
    import operator
    d_interClusters = {}  #the distances between all the clusteres
    z_keys = z.keys()
    n_keys = len(z_keys)
    for i in range(n_keys - 1):
        for j in range(i + 1, n_keys):
            delt = z[z_keys[i]] - z[z_keys[j]]
            d_interClusters[(z_keys[i], z_keys[j])] = sqrt(delt * delt.T)
    sorted_d = sorted(d_interClusters.iteritems(), key=operator.itemgetter(1))
    k = 0
    #if conditions satisfied, merge these clusters
    while k < L:
        if len(sorted_d) == 0:
            break
        if sorted_d[0][1] >= theta_d:
            break
        i = sorted_d[0][0][0]
        j = sorted_d[0][0][1]
        ni = len(labels[i])
        nj = len(labels[j])
        z[i] = (ni * z[i] + nj * z[j]) / float(ni + nj)
        del z[j]
        del sorted_d[0]
        index = 0
        while True:
            if index >= len(sorted_d):
                break
            if i in sorted_d[index][0] or j in sorted_d[index][0]:
                del sorted_d[index]
            else:
                index += 1
        k += 1
        print('One cluster has been merged because of getting too close to other clusters')
    return z    

def cluster_split(z, labels, X, c, theta_s, theta_n):
    """if conditions satisfied, start this splitting process"""
    d = {}  #the mean distance within each cluster
    d_mean = 0  #the mean distance of all samples
    for i in z.keys():
        delt = X[labels[i], :] - z[i]
        ni = len(labels[i])
        d[i] = sum(sqrt(sum(multiply(delt, delt), 1)), 0) / float(ni)
        d_mean += d[i] * float(ni)
    d_mean = d_mean / float(X.shape[0])
    splited = False
    k = 0.5 #0.5 can be changed to other values
    for i in z.keys():
        delt = X[labels[i], :] - z[i]
        sigma_i = sqrt(sum(multiply(delt, delt), 0) / float(len(labels[i])))
        j = sigma_i.argmax()
        if(sigma_i[0, j] > theta_s and ((d[i] > d_mean and len(labels[i]) > 2 * (theta_n + 1)) or len(z.keys()) <= c / 2.0)):
            max_index = max(z.keys())
            z[max_index+1] = z[i].copy()
            z[max_index+1][0, j] += k * sigma_i[0, j]
            z[i][0, j] -= k * sigma_i[0, j]
            print('One cluster has been splited because of too large standard deviation')
            splited = True
    return z, splited

def clusterByDistance(z, X):
    """assign each sample to the proper cluster to which the distance is the shortest"""
    labels = {}
    for i in z.keys():
        labels[i] = []
    for i in range(X.shape[0]):
        min_d = inf
        for j in z.keys():
            delt_ij = X[i, :] - z[j]
            d_ij = sqrt(delt_ij * delt_ij.T)
            if d_ij < min_d:
                min_d = d_ij
                min_j = j
        labels[min_j].append(i)
    return labels

#----------------------------------------------------------------------
def testIsodata():
    X = [[0, 0],
         [1, 1],
         [2, 2],
         [4, 3],
         [5, 3],
         [4, 4],
         [5, 4],
         [6, 5]]
    y = [0, 0, 0, 1, 1, 1, 1, 1]
    labels = isodata(X, c = 2, theta_n = 2, theta_s = 1, theta_d = 4, L= 1, I = 4, init_z= (0., 0.), ip = 1)
    plt.figure(1)
    for i in range(len(X)):
        #the original categories showed by different markers
        if y[i] == 0:
            mstr = '+'
        else:
            mstr = 'o'
        #clustering result showed by different colors
        if i in labels[labels.keys()[0]]:
            cstr = 'r'
        else:
            cstr = 'b'
        plt.scatter(X[i][0], X[i][1], color = cstr, marker = mstr, s = 50)
    plt.show()

testIsodata()    