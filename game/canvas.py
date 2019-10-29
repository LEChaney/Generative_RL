import numpy as np
import sys
import random
import pygame
import pygame.surfarray as surfarray
from pygame.locals import *
from itertools import cycle
import skimage

WIDTH_PIXELS  = 28  #use 256 (power of 2)
HEIGHT_PIXELS = 28
SCREEN_WIDTH = 128
SCREEN_HEIGHT = 128
BACKGROUND_COLOR = (0,0,0)
R = 1/np.cos(np.pi/4)

pygame.init()
FPSCLOCK = pygame.time.Clock()
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Canvas')

class GameState:
    def __init__(self, FPS):
        self.FPS = FPS
        self.surface = pygame.surface.Surface((WIDTH_PIXELS, HEIGHT_PIXELS))
        self.reset()

    def frame_step(self, input_actions):
        self.frame_count += 1

        x, y, width, height, i = (np.clip(input_actions[:-1], -1, 1) + 1) / 2
        x = x * WIDTH_PIXELS
        y = y * HEIGHT_PIXELS
        width = 1 + width * R * WIDTH_PIXELS
        height = 1 + height * R * HEIGHT_PIXELS
        r, g, b = 255 * np.array([i, i, i])
        orientation = input_actions[-1]
        orientation = orientation * 180
        
        paint_area = pygame.surface.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.ellipse(paint_area, (r, g, b), Rect(0, 0, width, height))
        paint_area = pygame.transform.rotate(paint_area, orientation)
        paint_dest = paint_area.get_rect(center=(x, y))
        self.surface.blit(paint_area, paint_dest)

        image_data = pygame.surfarray.array3d(self.surface)
        image_data = np.transpose(image_data, axes=[1, 0, 2])
        
        scaled = pygame.transform.scale(self.surface, [SCREEN_WIDTH, SCREEN_HEIGHT])
        SCREEN.blit(scaled, (0, 0))

        pygame.display.update()
        FPSCLOCK.tick(self.FPS)

        return image_data

    def get_current_frame(self):
        image_data = pygame.surfarray.array3d(self.surface)
        image_data = np.transpose(image_data, axes=[1, 0, 2])
        return image_data

    def reset(self):
        self.frame_count = 0
        self.surface.fill(BACKGROUND_COLOR)
        scaled = pygame.transform.scale(self.surface, [SCREEN_WIDTH, SCREEN_HEIGHT])
        SCREEN.blit(scaled, (0, 0))
        pygame.display.update()

