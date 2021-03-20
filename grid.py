"""
Grille
"""

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import typing


class Grid:
    """
    Grille
    """

    x: typing.List[int]
    y: typing.List[int]
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
        self.index = {}

        for y in range(self.nbRows):
            for x in range(self.nbCols):
                self.index[(y, x)] = x + y * self.nbCols

    def getIndex(self, row: int, col: int) -> int:
        """
        :param row: Ligne
        :param col: Colonne
        :return: Index associé
        """

        return self.index.get((row, col), -1)

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
            if idx != -1:
                N.append(idx)

        for dy in (-1, 1):
            idx = self.getIndex(y + dy, x)
            if idx != -1:
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

    def vectorToImage(self, V: np.matrix) -> np.matrix:
        """
        :param V: Matrice vecteur
        :return: Matrice image
        """

        img = np.zeros((self.nbRows, self.nbCols))
        K = self.index.keys()
        I = self.index.values()
        for k, idx in zip(K, I):
            img[k[0], k[1]] = V[idx]
        return np.asmatrix(img)

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

    def explicitEuler(self, U: typing.List[float], T: float, dt: float) -> typing.List[np.matrix]:
        """
        :param U: Matrice
        :param T: T
        :param dt: Delta t
        :return: Liste des matrices à chaque intervalle delta t
        """

        res = [U]
        t = 0
        Uk = U
        m_id = self.Identity()
        m_lap = self.Laplacian()

        while t < T:
            Uk1 = (m_id + dt * m_lap) * Uk
            t += dt
            res.append(Uk1)
            Uk = Uk1

        return res

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
