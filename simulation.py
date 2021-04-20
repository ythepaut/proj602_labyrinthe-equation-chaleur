import numpy as np
import tkinter as tk
import matplotlib
import matplotlib.cm
import typing
import math
import random

from grid import Grid


class Simulation:
    FRAMERATE: int = 60
    SPEED: float = 5.
    RADIUS: int = 5

    window: tk.Tk
    canvas: tk.Canvas

    ovalId: typing.Optional[int] = None
    imageId: typing.Optional[int] = None

    posX: float = 0.
    posY: float = 0.

    speedX: float = 0.
    speedY: float = 0.

    width: int
    height: int

    grid: Grid

    cmap: typing.Any

    background: tk.PhotoImage

    @classmethod
    def run(cls, grid: Grid, ratio: float):
        cls.grid = grid

        cls.width = int(grid.nbCols * ratio)
        cls.height = int(grid.nbRows * ratio)
        cls.ratio = ratio

        cls.window = tk.Tk()
        cls.canvas = tk.Canvas(cls.window, bg="white", width=cls.width, height=cls.height)

        cls.cmap = matplotlib.cm.get_cmap("twilight")

        print("Création de l'image")

        cls.background = tk.PhotoImage(width=cls.width, height=cls.height)
        for col in range(grid.nbCols):
            for row in range(grid.nbRows):
                idx = grid.getIndex(row, col)
                if idx is None:
                    color = (0, 0, 0)
                else:
                    dx, dy = grid.derivatives[idx]
                    angle = np.angle([complex(dx, dy)], deg=True)[0] + 180.
                    r, g, b, a = cls.cmap(angle / 360.)
                    color = (int(r * 255), int(g * 255), int(b * 255))

                pos = (col, row)
                cls.background.put("#%02x%02x%02x" % tuple(color), pos)

        print("Zoom de l'image")

        cls.background = cls.background.zoom(ratio, ratio)

        cls.canvas.bind("<Button-1>", cls.onClick)

        print("Affichage")

        cls.onClick(None)

        cls.canvas.pack()
        cls.update()
        cls.window.mainloop()

    @classmethod
    def onClick(cls, evt: typing.Optional):
        if evt is None:
            r = c = 0

            for i in range(100):
                r, c = random.randrange(cls.grid.nbRows), random.randrange(cls.grid.nbCols)

                if cls.grid.getIndex(r, c) is not None:
                    break

            cls.posX = (c / cls.grid.nbCols) * cls.width
            cls.posY = (r / cls.grid.nbRows) * cls.height
        else:
            cls.posX, cls.posY = evt.x, evt.y

        cls.speedX = cls.speedY = 0
        alpha = random.randrange(int(math.pi * 100)) / 100
        cls.speedY, cls.speedX = math.sin(alpha), math.cos(alpha)

    @classmethod
    def update(cls):
        cls.moveDot()

        if cls.imageId is not None:
            cls.canvas.delete(cls.imageId)

        cls.imageId = cls.canvas.create_image(0, 0, image=cls.background, anchor=tk.NW)

        if cls.ovalId is not None:
            cls.canvas.delete(cls.ovalId)

        cls.ovalId = cls.canvas.create_oval(
            cls.posX - cls.RADIUS,
            cls.posY - cls.RADIUS,
            cls.posX + cls.RADIUS,
            cls.posY + cls.RADIUS,
            fill="blue"
        )

        cls.window.after(math.ceil(1000. / cls.FRAMERATE), cls.update)

    @classmethod
    def moveDot(cls):
        row = (cls.posY / cls.height) * cls.grid.nbRows
        col = (cls.posX / cls.width) * cls.grid.nbCols

        idx = cls.grid.getIndex(int(row), int(col))
        if idx is not None:
            dx, dy = cls.grid.derivatives[idx]

            nx = cls.posX + cls.speedX * cls.SPEED
            ny = cls.posY + cls.speedY * cls.SPEED
            nrow = (ny / cls.height) * cls.grid.nbRows
            ncol = (nx / cls.width) * cls.grid.nbCols

            if cls.grid.getIndex(int(nrow), int(col)) is None:
                cls.speedY *= -1

            if cls.grid.getIndex(int(row), int(ncol)) is None:
                cls.speedX *= -1

            cls.posX += cls.speedX * cls.SPEED
            cls.posY += cls.speedY * cls.SPEED

            cls.speedX += dx / 5
            cls.speedY += dy / 5

            alpha = np.angle([complex(cls.speedX, cls.speedY)])[0]
            cls.speedY, cls.speedX = math.sin(alpha), math.cos(alpha)
