import numpy as np
import sklearn as sk
import scipy as sp
import scipy.spatial
import pylab as plot

np.random.seed(1)


def x2p(x=np.array([]), perplexity=40):
    # Get dimensions
    (n, d) = np.shape(x)
    distance = sp.spatial.distance.pdist(x, metric='euclidean')
    tri = np.zeros((n,n))
    tri[np.triu_indices(n, 1)] = distance
    tri = tri + tri.T
    np.fill_diagonal(tri, np.infty)
    P = np.zeros((n, n))
    maxit = 50
    sigma = np.ones(n)
    tol = 1e-5
    targetentropy = np.log2(perplexity)

    for i in range(n):
        sigmamin = -np.inf
        sigmamax = np.inf
        it = 1

        if i % 100 == 0:
            print("Computing conditional probabilities for point ", i, " of ", n, "...")
        p = np.exp(-np.square(tri[i]) / (2 * np.square(sigma[i]))) / (np.sum(np.exp(-np.square(tri[i]) / (2 * np.square(sigma[i])))))
        entropy = -sum(np.multiply(p[p != 0], np.log2(p[p != 0])))

        while abs(entropy-targetentropy) > tol and it < maxit:

            p = np.exp(-np.square(tri[i]) / (2 * np.square(sigma[i]))) / (
            np.sum(np.exp(-np.square(tri[i]) / (2 * np.square(sigma[i])))))

            entropy = -sum(np.multiply(p[0 != p], np.log2(p[p != 0])))

            if entropy > targetentropy:
                sigmamax = sigma[i].copy()
                if sigmamin == -np.inf:
                    sigma[i] = np.multiply(sigma[i], .5)
                else:
                    sigma[i] = np.multiply(np.add(sigma[i], sigmamin), 0.5)
            else:
                sigmamin = sigma[i].copy()
                if sigmamax == np.inf:
                    sigma[i] = np.multiply(sigma[i], 2)
                else:
                    sigma[i] = np.multiply(np.add(sigma[i], sigmamax), 0.5)
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
    plot.ion()
    x = pca(x, initial_dims).real
    (n, d) = x.shape
    maxit = 1000
    momentum_init = .5
    momentum_final = .8
    eta = 500
    min_gain = 0.01
    y = np.random.randn(n, intrinsic_dim)
    dY = np.zeros((n, intrinsic_dim))
    iY = np.zeros((n, intrinsic_dim))
    P = x2p(x, perplexity)
    P = np.divide(P + np.transpose(P), 2*n)
    P *= 4
    P = np.maximum(P, 1e-12)
    Q = np.zeros((n, n))
    gains = np.ones((n, intrinsic_dim))

    for iter in range(maxit):
        # Compute Q

        distance = sp.spatial.distance.pdist(y, metric='euclidean')
        tri = np.zeros((n, n))
        tri[np.triu_indices(n, 1)] = distance
        tri = tri + tri.T
        np.fill_diagonal(tri, np.infty)
        for j in range(n):
            q = np.divide(np.power(1 + np.square(tri[j]), -1), np.sum(np.power(1 + np.square(tri[j]), -1)))
            Q[j] = q
        Q = Q/np.sum(Q)
        Q = np.maximum(Q,1e-12)
        # Compute gradient
        PQ = P-Q

        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * np.power((1 + np.square(tri[:, i])), -1), (intrinsic_dim, 1)).T * (y[i, :]-y), 0)

        # Set new outputs
        if iter < 20:
            momentum = momentum_init
        else:
            momentum = momentum_final
        gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0))
        gains[gains < min_gain] = min_gain

        iY = momentum * iY - (eta * gains * dY)
        y += iY
        #y = y - np.tile(np.mean(y, 0), (n, 1))


        # Compute Current Cost
        C = np.sum(np.multiply(P, np.log(np.divide(P,Q))))


        if iter == 100:
            P = P/4

        if iter % 1 == 0:
            print("Iteration", (iter + 1), ": Error is ", round(C, 4))
            plot.clf()
            plot.pause(.001)
            plot.scatter(y[:, 0], y[:, 1], 20, labels)
            plot.pause(0.001)
    return y

if __name__ == "__main__":
    print ("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    print ("Running example on 2,500 MNIST digits...")
    x = np.loadtxt("mnist2500_X.txt")
    labels = np.loadtxt('mnist2500_labels.txt')
    y = tsne(x, 3, 30, 30)
    plot.scatter(y[:, 0], y[:, 1], 20, labels)
    plot.show()
