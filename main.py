from grid import Grid
from simulation import Simulation


THICKNESS = 10

START_TOKEN = 'a'
END_TOKEN = 'b'

labyrinth = [
    "a* *b",
    " * * ",
    " *   ",
    " * * ",
    " * * ",
    "   * ",
]

if __name__ == "__main__":
    G = Grid(len(labyrinth) * THICKNESS, len(labyrinth[0]) * THICKNESS)
    V = [0.] * G.size()

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

    V = G.explicitEuler(V, 2., 0.01, True)[-1]

    G.reloadValues(V)

    G.showValues()
    G.showDerivatives()

    Simulation.run(G, 10)
