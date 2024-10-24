import pygame
from pygame import gfxdraw
from constants import constants

class utils:
        
    # copy from gymnasium.envs.box2d.car_racing
    def draw_colored_polygon(surface, poly, color, zoom, translation, angle, clip = True):
        poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in poly]
        poly = [
            (c[0] * zoom + translation[0], c[1] * zoom + translation[1]) for c in poly
        ]
        # If not clip, chop out pixels not needed to enhance performance
        if not clip or any(
            (-constants.MAX_SHAPE_DIM <= coord[0] <= constants.VIDEO_WIDTH + constants.MAX_SHAPE_DIM)
            and (-constants.MAX_SHAPE_DIM <= coord[1] <= constants.VIDEO_HEIGHT  + constants.MAX_SHAPE_DIM)
            for coord in poly
        ):
            gfxdraw.aapolygon(surface, poly, color)
            gfxdraw.filled_polygon(surface, poly, color)

            