import argparse
import json
import typing

from grid import Grid
from settings import Settings
from simulation import Simulation


def init() -> typing.Tuple[typing.List[str], Settings]:
    """
    Init
    :return: Labyrinth and settings
    """

    argParser = argparse.ArgumentParser(description="Équation de la chaleur")
    argParser.add_argument("-s", "--settings", type=str, default="./settings1.json", help="Settings path")
    argParser.add_argument("-v", "--verbose", action="store_true", help="Verbose")
    argParser.add_argument("-a", "--animate", action="store_true", help="Animate heat progression")
    argParser.add_argument("-d", "--dirichlet", action="store_true", help="Use Dirichlet laplacian")
    opt, _ = argParser.parse_known_args()

    f = open(opt.settings, 'r')
    settingsDict = json.load(f)
    f.close()

    tokens = settingsDict.get("tokens", {})
    thickness = settingsDict.get("thickness", 5)
    labyrinth = settingsDict.get("labyrinth", [" "])

    zoom = settingsDict.get("zoom", None)
    if zoom is None:
        zoom = int(0.75 * 1080 / (thickness * len(labyrinth))) + 1

    settings = Settings(
        tokens.get("wall", ["*"]),
        tokens.get("start", ["@"]),
        tokens.get("empty", [" "]),
        thickness,
        zoom,
        settingsDict.get("eulerTime", 16.),
        settingsDict.get("dt", 0.01),
        settingsDict.get("ballSpeed", 5.),
        opt.animate,
        opt.dirichlet,
        opt.verbose
    )

    return labyrinth, settings


def main():
    labyrinth, settings = init()

    if settings.verbose:
        print("Création de la grille")

    maxLength = 0
    for row in labyrinth:
        if len(row) > maxLength:
            maxLength = len(row)

    G = Grid(len(labyrinth) * settings.thickness, maxLength * settings.thickness)
    V = [0.] * G.size()

    if settings.verbose:
        print("Remplissage de la grille")

    for row in range(len(labyrinth)):
        for col in range(len(labyrinth[row])):
            cell = labyrinth[row][col]
            if cell in settings.holeTokens:
                continue
            elif labyrinth[row][col] in settings.startTokens:
                idx = G.getIndex(row * settings.thickness + settings.thickness // 2,
                                 col * settings.thickness + settings.thickness // 2)
                V[idx] = 1e308
            elif cell in settings.wallTokens:
                for i in range(settings.thickness):
                    for j in range(settings.thickness):
                        G.unset(row * settings.thickness + i, col * settings.thickness + j)

    if settings.verbose:
        print("Calcul de Euler")

    useGif = settings.animate
    useDirichlet = settings.dirichlet
    V, arr = G.explicitEuler(V, settings.eulerTime, settings.dt, useDirichlet, useGif)

    if useGif:
        if settings.verbose:
            print("Export du GIF")

        G.showGif(arr)

    if settings.verbose:
        print("Calcul des dérivées")

    G.reloadValues(V)

    G.showValues()
    G.showDerivatives()

    Simulation.run(G, settings.zoom, settings)


if __name__ == "__main__":
    main()
