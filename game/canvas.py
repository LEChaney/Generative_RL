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

        x1, y1, x2, y2, l = (input_actions + 1) / 2
        # orientation = input_actions[-1]
        # orientation = orientation * 180
        x1 = int(np.clip(x1, 0, 1) * WIDTH_PIXELS)
        y1 = int(np.clip(y1, 0, 1) * HEIGHT_PIXELS)
        x2 = int(np.clip(x2, 0, 1) * WIDTH_PIXELS)
        y2 = int(np.clip(y2, 0, 1) * HEIGHT_PIXELS)
        # width = max(width * R * WIDTH_PIXELS, 1)
        # height = max(height * R * HEIGHT_PIXELS, 1)
        r, g, b = 255 * np.clip((l, l, l), 0, 1)
        
        pygame.draw.line(self.surface, (r, g, b), (x1, y1), (x2, y2))
        # pygame.draw.rect(self.surface, (255, 255, 255), Rect(x, y, 1, 1))
        # paint_area = pygame.surface.Surface((width, height), pygame.SRCALPHA)
        # pygame.draw.ellipse(paint_area, (r, g, b), Rect(0, 0, width, height))
        # paint_area = pygame.transform.rotate(paint_area, orientation)
        # paint_dest = paint_area.get_rect(center=(x, y))
        # self.surface.blit(paint_area, paint_dest)

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

    # New state expects an un-normalized numpy array (values ranging from 0 - 255), shape: [H x W X C]
    # Will be reset using BACKGROUND_COLOR if no new state is specified
    def reset(self, new_state = None):
        if new_state is not None:
            if new_state.shape[-1] < 3:
                new_state = np.tile(new_state, [1, 1, 3])
                new_state = np.transpose(new_state, axes=[1, 0, 2])
            self.surface = pygame.surfarray.make_surface(new_state)
        else:
            self.surface.fill(BACKGROUND_COLOR)

        self.frame_count = 0
        scaled = pygame.transform.scale(self.surface, [SCREEN_WIDTH, SCREEN_HEIGHT])
        SCREEN.blit(scaled, (0, 0))
        pygame.display.update()

