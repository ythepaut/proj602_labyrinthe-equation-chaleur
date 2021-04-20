from grid import Grid
from simulation import Simulation


THICKNESS = 15
ZOOM = 6
EULER_TIME = 16.

START_TOKEN = '@'
END_TOKEN = '#'

labyrinth = [
    "@  * *   ",
    " * * * * ",
    " * *   * ",
    " * * *** ",
    " *   *   ",
    "** * * **",
    "   * * * ",
    " ***** * ",
    "     * @ ",
]

if __name__ == "__main__":
    print("Création de la grille")

    G = Grid(len(labyrinth) * THICKNESS, len(labyrinth[0]) * THICKNESS)
    V = [0.] * G.size()

    print("Remplissage de la grille")

    for row in range(len(labyrinth)):
        for col in range(len(labyrinth[row])):
            if labyrinth[row][col] in (' ', END_TOKEN):
                continue
            elif labyrinth[row][col] == START_TOKEN:
                idx = G.getIndex(row * THICKNESS + THICKNESS // 2, col * THICKNESS + THICKNESS // 2)
                V[idx] = 1.
            else:
                for i in range(THICKNESS):
                    for j in range(THICKNESS):
                        G.unset(row * THICKNESS + i, col * THICKNESS + j)

    print("Calcul de Euler")

    V = G.explicitEuler(V, EULER_TIME, 0.01, True)[-1]

    print("Calcul des dérivées")

    G.reloadValues(V)

    G.showValues()
    G.showDerivatives()

    Simulation.run(G, ZOOM)
