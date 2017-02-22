import numpy as np
import sklearn as sk
import scipy as sp
import scipy.spatial


def x2p(x=np.array([]), perplexity=40):
    # Get dimensions
    (n, d) = np.shape(x)
    distance = sp.spatial.distance.pdist(x, metric='euclidean')
    tri = np.zeros((n,n))
    tri[np.triu_indices(n, 1)] = distance
    tri[np.tril_indices(n, -1)] = distance
    np.fill_diagonal(tri, np.infty)
    P = np.zeros((n, n))
    maxit = 100
    sigma = np.ones(n)
    tol = 1e-5
    targetentropy = np.log2(perplexity)

    for i in range(n):
        sigmamin = -np.inf
        sigmamax = np.inf
        it = 1

        if i % 100 == 0:
            print ("Computing conditional probabilities for point ", i, " of ", n, "...")
        p = np.exp(-np.square(tri[i]) / (2 * sigma[i])) / (np.sum(np.exp(-np.square(tri[i]) / (2 * sigma[i]))))
        entropy = -sum(np.multiply(p[p != 0], np.log2(p[p != 0])))

        while abs(entropy-targetentropy > tol) and it < maxit:

            p = np.exp(-np.square(tri[i])/(2*sigma[i]))/(np.sum(np.exp(-np.square(tri[i])/(2*sigma[i]))))
            entropy = -sum(np.multiply(p[0 != p], np.log2(p[p != 0])))

            if entropy > targetentropy:
                sigmamax = sigma[i].copy()
                if sigmamin == -np.inf:
                    sigma = np.multiply(sigma, 0.5)
                else:
                    sigma = np.multiply(np.add(sigma, sigmamin), 0.5)
            else:
                sigmamin = sigma[i].copy()
                if sigmamax == np.inf:
                    sigma = np.multiply(sigma, 2)
                else:
                    sigma = np.multiply(np.add(sigma, sigmamax), 2)
            it += 1
        P[i] = p
    return P


def pca(x=np.array([]), output_dim=1):
    (n, d) = np.shape(x)
    x -= np.tile(np.mean(x, 0), (n, 1))
    (l, m) = np.linalg.eig(np.dot(x.T, x))
    x_new = np.dot(x, m[:, 0:output_dim])
    return x_new


# Set parameters, initiaize Y, set Q
def tsne(x=np.array([]), intrinsic_dim = 2, perplexity = 30):
    x = pca(x, 30).real
    (n, d) = x.shape
    maxit = 1000
    momentum_init = .5
    momentum_final = .8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, intrinsic_dim)
    dY = np.zeros((n, intrinsic_dim))
    iY = np.zeros((n, intrinsic_dim))
    P = x2p(x, perplexity)

if __name__ == "__main__":
    print ("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    print ("Running example on 2,500 MNIST digits...")
    x = np.loadtxt("mnist2500_X.txt")
