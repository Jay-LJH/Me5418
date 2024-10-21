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
        if not clip or any(
            (-constants.max_shape_dim <= coord[0] <= constants.video_width + constants.max_shape_dim)
            and (-constants.max_shape_dim <= coord[1] <= constants.video_height  + constants.max_shape_dim)
            for coord in poly
        ):
            gfxdraw.aapolygon(surface, poly, color)
            gfxdraw.filled_polygon(surface, poly, color)