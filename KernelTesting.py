import numpy as np
import pylab as plot

x1 = np.zeros((500,1))
x2 = np.arange(0,10,.02)


y = np.sqrt((np.square(x2-x1)))
out = np.exp(-np.square(y))

def rbf():
    out = np.exp(-np.square(y))
    return out

def tdist(dof=1):1
    out = np.power(1+np.square(y),-dof)
    return out

def matern32():
    out = (1 + np.sqrt(3)*y)* np.exp(-np.sqrt(3)*y)
    return out


plot.scatter(y, tdist(2), c='red', s=2)
plot.scatter(y, tdist(1), c='green', s=2)
plot.scatter(y, rbf(), c='blue', s=2)
plot.show()