from numpy import *

#----------------------------------------------------------------------
def h_cluster(X, c):
    n_samples = X.shape[0]
    n_clusters = n_samples
    labels = {}
    m = {}
    for i in range(n_samples):
        labels[i] = [i]
        m[i] = X[i]
    while True:
        if n_clusters == c:
            break
        min_dis = inf
        index_pair = zeros((c))
        for i in labels.keys():
            ni = len(labels[i])
            for j in labels.keys():
                if i >= j:
                    continue
                nj = len(labels[j])
                dis = ni * nj / float(ni + nj) * sum((m[i] - m[j]) *(m[i] - m[j]))
                if dis < min_dis:
                    min_dis = dis
                    index_pair[0] = i
                    index_pair[1] = j
        labels[index_pair[0]].extend(labels[index_pair[1]])
        del labels[index_pair[1]]
        print('One cluster merged')
        m[index_pair[0]] = sum(X[labels[index_pair[0]], :], 0) / float(len(labels[index_pair[0]]))
        del m[index_pair[1]]
        n_clusters -= 1
    return labels

def test_h_cluster():
    import matplotlib.pyplot as plt
    from time import clock    
    N = 50
    X1 = random.randn(N, 2) + array([8, 10])#two-dimensional normal samples
    X2 = random.randn(N, 2) + array([6, 8])
    X3 = random.randn(N, 2) + array([5, 11])
    X = vstack((X1, X2, X3))
    start = clock()
    labels = h_cluster(X, 3)
    print('The clustering process used %.2fs ' % (clock() - start))
    plt.figure(1)
    ax = plt.subplot(111)
    for i in range(3 * N):
        #the original categories showed by different markers
        if i < N:
            mstr = '+'
        elif i < 2 * N:
            mstr = 's'
        else:
            mstr = 'o'
        #clustering result showed by different colors
        if i in labels[labels.keys()[0]]:
            cstr = 'r'
        elif i in labels[labels.keys()[1]]:
            cstr = 'g'
        else:
            cstr = 'b'
        ax.scatter(X[i, 0], X[i, 1], color = cstr, marker = mstr, s = 40)
    plt.show()

test_h_cluster()
    
                
