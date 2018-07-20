
import numpy as np
from sklearn.gaussian_process import GaussianProcess


def gpTest1():
    xx = np.array([[10.]])
    X = np.array([[1., 3., 5., 6., 7., 8., 9.]]).T
    y = (X * np.sin(X)).ravel()
    gp = GaussianProcess(theta0=0.1, thetaL=.001, thetaU=1.)
    gp.fit(X, y)
    p = gp.predict(xx)
    print(X, y)
    print(p, xx*np.sin(xx))

if __name__ == '__main__':
    gpTest1()