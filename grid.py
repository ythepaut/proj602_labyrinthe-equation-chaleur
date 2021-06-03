"""
Grille
"""

import numpy as np
from scipy.sparse import *
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import typing
import math


class Grid:
    """
    Grille
    """

    x: typing.List[int]
    y: typing.List[int]
    values: typing.List[float]
    derivatives: typing.List[typing.Tuple[float, float]]
    index: typing.Dict[typing.Tuple[int, int], int]
    nbCols: int
    nbRows: int

    def __init__(self, nbRows: int, nbCols: int):
        self.nbCols = nbCols
        self.nbRows = nbRows
        self.number()

    def number(self) -> None:
        """
        Number
        """

        self.x = [j for _ in range(self.nbRows) for j in range(self.nbCols)]
        self.y = [i for i in range(self.nbRows) for _ in range(self.nbCols)]
        self.values = [0. for _ in range(self.nbRows) for _ in range(self.nbCols)]
        self.derivatives = [(0., 0.) for _ in range(self.nbRows) for _ in range(self.nbCols)]
        self.index = {}

        for y in range(self.nbRows):
            for x in range(self.nbCols):
                self.index[(y, x)] = x + y * self.nbCols

    def getIndex(self, row: int, col: int) -> typing.Optional[int]:
        """
        :param row: Ligne
        :param col: Colonne
        :return: Index associé
        """

        return self.index.get((row, col), None)

    def unset(self, row: int, col: int):
        if (row, col) in self.index:
            self.index.pop((row, col))

    def neighbors(self, idx: int) -> typing.List[int]:
        """
        :param idx: Indice
        :return: Indices voisins
        """

        N = []
        y = self.y[idx]
        x = self.x[idx]

        for dx in (-1, 1):
            idx = self.getIndex(y, x + dx)
            if idx is not None:
                N.append(idx)

        for dy in (-1, 1):
            idx = self.getIndex(y + dy, x)
            if idx is not None:
                N.append(idx)

        return N

    def size(self) -> int:
        """
        :return: Taille
        """

        return len(self.x)

    def Identity(self) -> sparse.csc_matrix:
        """
        :return: Matrice identité
        """

        R = []  # les lignes des coefficients
        C = []  # les colonnes des coefficients
        V = []  # les valeurs des coefficients
        for idx in self.index.values():
            R.append(idx)
            C.append(idx)
            V.append(1.0)
        M = sparse.coo_matrix((V, (R, C)), shape=(self.size(), self.size()))
        return M.tocsc()

    def Laplacian(self) -> sparse.csc_matrix:
        """
        :return: Laplacien Neumann
        """

        R = []
        C = []
        V = []

        for idx in self.index.values():
            neighbors = self.neighbors(idx)

            R.append(idx)
            C.append(idx)
            V.append(-1. * len(neighbors))

            for col in neighbors:
                R.append(idx)
                C.append(col)
                V.append(1.)

        M = sparse.coo_matrix((V, (R, C)), shape=(self.size(), self.size()))
        return M.tocsc()

    def LaplacianD(self) -> sparse.csc_matrix:
        """
        :return: Laplacien Dirichlet
        """

        R = []
        C = []
        V = []

        for idx in self.index.values():
            neighbors = self.neighbors(idx)

            R.append(idx)
            C.append(idx)
            V.append(-4.)

            for col in neighbors:
                R.append(idx)
                C.append(col)
                V.append(1.)

        M = sparse.coo_matrix((V, (R, C)), shape=(self.size(), self.size()))
        return M.tocsc()

    def vectorToImage(self, V: np.matrix, raw: bool = True) \
            -> typing.Tuple[np.matrix, float, float]:
        """
        :param V: Matrice vecteur
        :return: Matrice image
        :param raw: Raw or processed
        """

        img = np.zeros((self.nbRows, self.nbCols))
        K = self.index.keys()
        I = self.index.values()

        mini = None
        maxi = None

        if not raw:
            for _, idx in zip(K, I):
                if V[idx] != 0:
                    if mini is None or mini > V[idx]:
                        mini = V[idx]

                    if maxi is None or maxi < V[idx]:
                        maxi = V[idx]

            if mini is None:
                mini = 0.
            else:
                mini = math.log(mini)

            if maxi is None:
                maxi = 100.
            else:
                maxi = math.log(maxi)

        for k, idx in zip(K, I):
            if raw:
                img[k[0], k[1]] = V[idx]
            else:
                if V[idx] == 0.:
                    log = 0
                    # Because walls are set to 0, we must not have any negative number,
                    #  else they will start to glow as if they were hot
                else:
                    log = math.log(V[idx]) - mini
                img[k[0], k[1]] = log

        return np.asmatrix(img), 0, maxi - mini

    @staticmethod
    def imageToVector(img: np.matrix) -> typing.List[int]:
        """
        :param img: Matrice image
        :return: Liste vecteur
        """

        values = []

        for row in img:
            for value in row:
                values.append(value)

        return values

    def explicitEuler(self,
                      U: typing.List[float],
                      T: float,
                      dt: float,
                      dirichlet: bool,
                      outputList: bool) \
            -> typing.Tuple[np.matrix, typing.List[np.matrix]]:
        """
        :param U: Matrice
        :param T: T
        :param dt: Delta t
        :param dirichlet: Laplacien Dirichlet ou Neumann
        :param outputList: Outputs a list or not
        :return: Liste des matrices à chaque intervalle delta t
        """

        if outputList:
            arr = [U]
        else:
            arr = []

        t = 0
        Uk = U
        m_id = self.Identity()

        if dirichlet:
            m_lap = self.LaplacianD()
        else:
            m_lap = self.Laplacian()

        while t < T:
            Uk1 = (m_id + dt * m_lap) * Uk
            t += dt

            if outputList:
                arr.append(Uk1)

            Uk = Uk1

        return Uk, arr

    def implicitEuler(self,
                      U: typing.List[float],
                      T: float,
                      dt: typing.Optional[float]) \
            -> typing.Union[typing.List[np.matrix], np.matrix]:
        """
        :param U: Matrice
        :param T: T
        :param dt: Delta T, ou None pour avoir directement au temps T
        :return: Liste des matrices à chaque intervalle delta t, ou directement la matrice
        """

        if dt == 0.:
            raise ValueError("Mauvais dt")

        if dt is None:
            Uk = U * self.Identity()
            A = self.Identity() - T * self.Laplacian()
            B = linalg.splu(A)
            return B.solve(Uk)
        else:
            res = [U]
            t = 0
            Uk = U * self.Identity()
            A = self.Identity() - dt * self.Laplacian()
            B = linalg.splu(A)

            while t < T:
                Uk1 = B.solve(Uk)
                t += dt
                res.append(Uk1)
                Uk = Uk1

            return res

    def showValues(self):
        res: np.matrix = np.asmatrix(np.zeros((self.nbRows, self.nbCols)))
        values: typing.List[float] = self.values[:]
        values.sort()

        maxi = values[-1]
        maxi_log = math.log(maxi)

        mini = -100
        for x in values:
            if x != 0.:
                mini = math.log(x)
                break  # We're sorted so we can break

        for row in range(self.nbRows):
            for col in range(self.nbCols):
                idx = self.getIndex(row, col)
                if idx is None:
                    res[row, col] = (mini - maxi_log) * 1.1
                else:
                    value = self.values[idx] / maxi
                    if value == 0:
                        res[row, col] = (mini - maxi_log) * 1.1
                    else:
                        res[row, col] = math.log(value)

        plt.imshow(res, cmap="magma")
        plt.show()

    def showDerivatives(self):
        res: np.matrix = np.asmatrix(np.zeros((self.nbRows, self.nbCols)))

        for row in range(self.nbRows):
            for col in range(self.nbCols):
                idx = self.getIndex(row, col)
                if idx is None:
                    res[row, col] = 0.
                elif self.derivatives[idx] == (0., 0.):
                    res[row, col] = 0.
                else:
                    x, y = self.derivatives[idx]
                    res[row, col] = np.angle([complex(x, y)])[0]

        plt.imshow(res, cmap="twilight")
        plt.show()

    def reloadValues(self, V: np.matrix):
        # Update values with "Euler-ed" matrix
        for row in range(self.nbRows):
            for col in range(self.nbCols):
                idx = self.getIndex(row, col)
                if idx is not None:
                    self.values[idx] = V[idx]

        # Derivative
        for row in range(self.nbRows):
            for col in range(self.nbCols):
                idx = self.getIndex(row, col)
                if idx is None:
                    continue

                idx_xm1 = self.getIndex(row - 1, col)
                idx_xp1 = self.getIndex(row + 1, col)
                idx_yp1 = self.getIndex(row, col + 1)
                idx_ym1 = self.getIndex(row, col - 1)

                dx = 0.
                dy = 0.

                if idx_xm1 is not None:
                    dx += - self.values[idx_xm1] + self.values[idx]

                if idx_xp1 is not None:
                    dx += - self.values[idx] + self.values[idx_xp1]

                if idx_yp1 is not None:
                    dy += - self.values[idx_yp1] + self.values[idx]

                if idx_ym1 is not None:
                    dy += - self.values[idx] + self.values[idx_ym1]

                if (dx, dy) == (0., 0.):
                    self.derivatives[idx] = (0., 0.)
                else:
                    alpha = np.angle([complex(dx, dy)])[0]
                    self.derivatives[idx] = (-math.sin(alpha), math.cos(alpha))
                    # NOTE: I'm not sure if we should take the "true" derivative or its orthogonal
                    #       vector, so I used the orthogonal one, pointing toward the origin

    def showGif(self, mats: typing.List[np.matrix]) -> None:
        fig = plt.figure()

        imgs = []

        for mat in mats:
            values, mini, maxi = self.vectorToImage(mat, False)
            imgs.append([plt.imshow(values, cmap="hot", vmin=mini, vmax=maxi)])

        ani = animation.ArtistAnimation(fig, imgs, interval=50, blit=False, repeat_delay=0)
        writer = animation.PillowWriter(fps=20)
        ani.save("chaleur.gif", writer=writer)


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


def show(M: csc_matrix) -> None:
    res: np.matrix = np.asmatrix(np.zeros(M.shape))

    for row in range(res.shape[0]):
        for col in range(res.shape[1]):
            res[row, col] = M[row, col]

    plt.imshow(res)
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
