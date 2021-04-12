import numpy as np
import pygame
import pygame.gfxdraw
import matplotlib
import matplotlib.cm
import typing
import math
import random

from grid import Grid


class Simulation:
    SPEED: float = 5.

    clock: pygame.time.Clock = pygame.time.Clock()
    window: pygame

    posX: float = 0.
    posY: float = 0.

    speedX: float = 0.
    speedY: float = 0.

    width: int
    height: int

    grid: Grid

    cmap: typing.Any

    background: typing.Dict

    @classmethod
    def run(cls, grid: Grid, ratio: float):
        pygame.init()

        cls.grid = grid

        cls.width = int(grid.nbCols * ratio)
        cls.height = int(grid.nbRows * ratio)

        cls.window = pygame.display.set_mode((cls.width, cls.height))

        cls.cmap = matplotlib.cm.get_cmap("twilight")

        cls.background = {(r, c): 0 for r in range(grid.nbRows) for c in range(grid.nbCols)}
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

                cls.background[(row, col)] = color

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            for col in range(grid.nbCols):
                for row in range(grid.nbRows):
                    rect = pygame.Rect((col * ratio, row * ratio), (ratio, ratio))
                    pygame.draw.rect(cls.window, cls.background[(row, col)], rect)

            pygame.gfxdraw.filled_circle(cls.window, int(cls.posX), int(cls.posY), 5, (255, 0, 0))

            if pygame.mouse.get_pressed()[0]:
                cls.posX, cls.posY = pygame.mouse.get_pos()
                cls.speedX = cls.speedY = 0
                cls.speedX = (random.randint(0, 100) - 50) / 100
                cls.speedY = (random.randint(0, 100) - 50) / 100

            cls.moveDot()

            pygame.display.flip()
            cls.clock.tick(60)

        pygame.quit()

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
