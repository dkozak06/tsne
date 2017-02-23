import numpy as np
import sklearn as sk
import scipy as sp
import scipy.spatial
import pylab as plot


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
            print("Computing conditional probabilities for point ", i, " of ", n, "...")
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


def pca(x=np.array([]), output_dim=30):
    (n, d) = np.shape(x)
    x -= np.tile(np.mean(x, 0), (n, 1))
    (l, m) = np.linalg.eig(np.dot(x.T, x))
    x_new = np.dot(x, m[:, 0:output_dim])
    return x_new


# Set parameters, initialize Y, set Q
def tsne(x=np.array([]), intrinsic_dim = 2, initial_dims= 30,  perplexity = 30):
    x = pca(x, initial_dims).real
    (n, d) = x.shape
    maxit = 1000
    momentum_init = .5
    momentum_final = .8
    eta = 500
    y = np.random.randn(n, intrinsic_dim)
    dY = np.zeros((n, intrinsic_dim))
    dYold = np.zeros((n, intrinsic_dim))
    P = x2p(x, perplexity)
    P = np.divide(P + np.transpose(P), 2*n)
    P = np.maximum(P, 1e-12)
    Q = np.zeros((n, n))

    for iter in range(maxit):
        # Compute Q

        distance = sp.spatial.distance.pdist(y, metric='euclidean')
        tri = np.zeros((n, n))
        tri[np.triu_indices(n, 1)] = distance
        tri[np.tril_indices(n, -1)] = distance
        np.fill_diagonal(tri, np.infty)
        for j in range(n):
            q = np.exp(-np.square(tri[j]))/np.sum(np.exp(-np.square(tri[j])))
            Q[j] = q

        # Compute gradient
        PQ = P-Q
        for i in range(n):
            dY = np.multiply(2, np.sum(np.multiply(PQ[i]-np.transpose(PQ)[i], y[i, :]-y)))

        # Set new outputs
        if iter < 20:
            momentum = momentum_init
        else:
            momentum = momentum_final

        y += eta * dY + momentum * (dY - dYold)
        dYold = dY

        # Compute Current Cost
        C = np.sum(np.multiply(P, np.log(np.divide(P,Q))))
        print("Iteration", (iter + 1), ": Error is ", round(C, 2))

        if iter == 100:
            P = P/4
    return y

if __name__ == "__main__":
    print ("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    print ("Running example on 2,500 MNIST digits...")
    x = np.loadtxt("mnist2500_X.txt")
    labels = np.loadtxt('mnist2500_labels.txt')
    y = tsne(x, 2, 50, 60)
    plot.scatter(y[:, 0], y[:, 1], 20, labels)
    plot.show()
