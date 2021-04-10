import numpy as np
from scipy.sparse import *
import scipy.sparse.linalg as linalg
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import typing
import math

from grid import Grid


def show(M: csc_matrix) -> None:
    res: np.matrix = np.asmatrix(np.zeros(M.shape))

    for row in range(res.shape[0]):
        for col in range(res.shape[1]):
            res[row, col] = M[row, col]

    plt.imshow(res)
    plt.show()


def showGif(mats: typing.List[np.matrix]) -> None:
    fig = plt.figure()

    imgs = []
    for mat in mats:
        imgs.append([plt.imshow(D.vectorToImage(mat), cmap="hot", vmin=0.0, vmax=1.0)])

    ani = animation.ArtistAnimation(fig, imgs, interval=50, blit=False, repeat_delay=0)
    # writer = animation.PillowWriter(fps=20)
    # ani.save("II-1-diffusion.gif", writer=writer)
    plt.show()


def diffuseImageRGB(img, T, verbose=True):
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    D = Grid(*r.shape)
    res = img.copy()

    if verbose:
        print("Diffusion RGB: ", end='')

    for i in range(3):
        if verbose:
            print("RGB"[i] + "...", end='')

        vect = D.imageToVector((r, g, b)[i])
        frame = D.implicitEuler(vect, T, None)
        res[:, :, i] = D.vectorToImage(frame)

    if verbose:
        print("OK")

    return res


def EQM(u: np.matrix, g: np.matrix) -> float:
    res = 0.

    for row in range(u.shape[0]):
        for col in range(u.shape[1]):
            res += abs(u[row, col] - g[row, col]) ** 2

    res /= (u.shape[0] * u.shape[1])
    return res


def PSNR_RGB(u, g):
    eqm = 0.
    for i in range(3):
        eqm += EQM(u[:, :, i], g[:, :, i])
    eqm /= 3.

    v = 0.
    for row in range(u.shape[0]):
        for col in range(u.shape[1]):
            for i in range(3):
                if u[row, col, i] > v:
                    v = u[row, col, i]

                if g[row, col, i] > v:
                    v = g[row, col, i]

    if eqm == 0.:
        return math.inf
    else:
        return 10 * math.log(v ** 2 / eqm, 10)


if __name__ == "__main__":
    D = Grid(50, 100)

    """
    for i in range(10):
        for j in range(20):
            D.unset(20 + i, 30 + j)
    """

    """
    for i in range(-5, 5):
        for j in range(-5, 5):
            if i*i + j*j < 25:
                D.unset(20 + i, 70 + j)
    """

    for i in range(5):
        for j in range(20):
            D.unset(20 + i, 30 + j)
            D.unset(20 + j, 30 + i)

    V = [0.] * D.size()
    V[D.getIndex(10, 10)] = 1.

    V = D.explicitEuler(V, 5.0, 0.01)[-1]

    D.reloadValues(V)

    D.showValues()
    D.showDerivatives()

if False and __name__ == "__main__":
    print("I.1")
    print("Classe Grid")

    D = Grid(3, 4)
    assert D.x == [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    assert D.y == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    assert D.index == {(0, 0): 0, (0, 1): 1, (0, 2): 2, (0, 3): 3, (1, 0): 4, (1, 1): 5, (1, 2): 6,
                       (1, 3): 7, (2, 0): 8, (2, 1): 9, (2, 2): 10, (2, 3): 11}

    ##################################################

    print("\nI.2")
    print("Voisins")

    assert D.neighbors(0) == [1, 4]
    assert D.neighbors(5) == [4, 6, 1, 9]

    ##################################################

    print("\nI.3")
    print("Opérateur Laplacien")

    D = Grid(3, 3)
    show(D.Laplacian())
    show(D.LaplacianD())

    ##################################################

    print("\nII.1")
    print("Diffusion simple par Euler explicite")

    D = Grid(20, 20)
    V = [0.] * D.size()
    V[42] = 5.0
    V[250] = 10.0

    showGif(D.explicitEuler(V, 8.0, 0.2))

    showGif(D.explicitEuler(V, 8.0, 0.025))
    showGif(D.explicitEuler(V, 8.0, 0.3))

    ##################################################

    print("\nII.2")
    print("Diffusion simple par Euler implicite")

    D = Grid(20, 20)
    V = [0.] * D.size()
    V[42] = 5.0
    V[250] = 10.0

    xVals = np.arange(0., 8., 0.2)
    imgs = []

    for t in xVals:
        imgs.append(D.implicitEuler(V, t, None))

    print("Directement à T :")
    showGif(imgs)

    print("En passant par un delta T :")
    showGif(D.implicitEuler(V, 8., 0.2))

    ##################################################

    print("\nII.3")
    print("Diffusion sur une image")

    imgNoisy = mpimg.imread("Images/mandrill-240-b01.png")
    plt.imshow(diffuseImageRGB(imgNoisy, 1.))
    plt.show()

    ##################################################

    print("\nII.4")
    print("Diffusion comme minimisation d'une fonctionnelle")

    imgOrig = mpimg.imread("Images/mandrill-240.png")
    assert PSNR_RGB(imgOrig, imgOrig) == math.inf

    dt = 0.05
    xVals = np.arange(0., 1., dt)
    yPSNR = []
    bestPSNR = -math.inf
    best_t = 0.
    cpt = 0

    for t in xVals:
        psnr = PSNR_RGB(imgOrig, diffuseImageRGB(imgNoisy, t, verbose=False))
        yPSNR.append(psnr)

        if psnr > bestPSNR:
            bestPSNR = psnr
            best_t = t

        cpt += 1
        print("\rCalcul en cours,", int(cpt * 100 / len(xVals)), "%", end="")
    print()

    ax = plt.subplot()
    ax.plot(xVals, yPSNR, color="red")
    plt.show()

    print("Meilleur T :", int(best_t * 1000) / 1000, "s")

    plt.imshow(diffuseImageRGB(imgNoisy, best_t))
    plt.show()
