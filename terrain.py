import numpy as np
import pygame as pg

from camera import world_to_screen


class Terrain:

    def __init__(self, oceans: list[tuple[int, int]],
                 ocean_color: tuple[int, int, int] = (0, 0, 255),
                 ground_color: tuple[int, int, int] = (50, 200, 50)) -> None:
        """Initialize the terrain with specified oceans and colors

        Args:
            oceans (list[tuple[int, int]]): (start, end) tuples for ocean regions
            ocean_color (tuple[int, int, int], optional): ocean color. Defaults to (0, 0, 255).
            ground_color (tuple[int, int, int], optional): ground color. Defaults to (50, 200, 50).
        """
        self.oceans: list[tuple[int, int]] = oceans
        self.ocean_color: tuple[int, int, int] = ocean_color
        self.ground_color: tuple[int, int, int] = ground_color

    def is_ocean(self, x: int) -> bool:
        """Whether the given x-coordinate is in an ocean region

        Args:
            x (int): x-coordinate (world position)

        Returns:
            bool: whether the x-coordinate is in an ocean region
        """
        return any(start <= x <= end for start, end in self.oceans)
    
    def draw(self, screen: pg.Surface, camera_pos: np.ndarray) -> None:
        """Draw the terrain on the screen

        Args:
            screen (pg.Surface): PyGame screen
            camera_pos (np.ndarray): camera position in world coordinates
        """
        # Draw basic ground
        y_ground = world_to_screen(np.array([0.0, 0.0]), camera_pos, screen.get_size())[1]
        pg.draw.rect(screen, self.ground_color,
                     (0, y_ground, screen.get_width(), screen.get_height() - y_ground))
        
        # Draw oceans (if on screen)
        for start, end in self.oceans:
            if start < camera_pos[0] + screen.get_width() and end > camera_pos[0]:
                x_start = max(start - camera_pos[0], 0)
                x_end = min(end - camera_pos[0], screen.get_width())
                pg.draw.rect(screen, self.ocean_color,
                             (x_start, y_ground, x_end - x_start, screen.get_height() - y_ground))
