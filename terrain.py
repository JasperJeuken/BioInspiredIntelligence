"""Terrain layout"""
import numpy as np
import pygame as pg

from camera import world_to_screen


class Terrain:

    def __init__(self,
                 oceans: list[tuple[int, int]],
                 runways: list[tuple[int, int]],
                 ocean_color: tuple[int, int, int] = (0, 0, 255),
                 runway_color: tuple[int, int, int] = (100, 100, 100),
                 ground_color: tuple[int, int, int] = (50, 200, 50)) -> None:
        """Initialize the terrain with specified oceans and colors

        Args:
            oceans (list[tuple[int, int]]): (start, end) tuples for ocean regions
            runways (list[tuple[int, int]]): (start, end) tuples for runway regions
            ocean_color (tuple[int, int, int], optional): ocean color. Defaults to (0, 0, 255)
            runway_color (tuple[int, int, int], optional): runway color. Defaults to (200, 200, 200)
            ground_color (tuple[int, int, int], optional): ground color. Defaults to (50, 200, 50)
        """
        self.oceans: list[tuple[int, int]] = oceans
        self.runways: list[tuple[int, int]] = runways

        self.ocean_color: tuple[int, int, int] = ocean_color
        self.runway_color: tuple[int, int, int] = runway_color
        self.ground_color: tuple[int, int, int] = ground_color

    def is_ocean(self, x: int) -> bool:
        """Whether the given x-coordinate is in an ocean region

        Args:
            x (int): x-coordinate (world position)

        Returns:
            bool: whether the x-coordinate is in an ocean region
        """
        return any(start <= x <= end for start, end in self.oceans)
    
    def _draw_collection(self, screen: pg.Surface, camera_pos: np.ndarray,
                         collection: list[tuple[int, int]], color: tuple[int, int, int]) -> None:
        """Draw a collection of regions on the screen

        Args:
            screen (pg.Surface): PyGame screen
            camera_pos (np.ndarray): camera position in world coordinates
            collection (list[tuple[int, int]]): collection to draw
            color (tuple[int, int, int]): color to use for drawing
        """
        y_ground = world_to_screen(np.array([0.0, 0.0]), camera_pos, screen.get_size())[1]
        for start, end in collection:
            if start < camera_pos[0] + screen.get_width() and end > camera_pos[0]:
                x_start = max(start - camera_pos[0], 0)
                x_end = min(end - camera_pos[0], screen.get_width())
                pg.draw.rect(screen, color,
                             (x_start, y_ground, x_end - x_start, screen.get_height() - y_ground))
    
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
        
        # Draw oceans and runways
        self._draw_collection(screen, camera_pos, self.oceans, self.ocean_color)
        self._draw_collection(screen, camera_pos, self.runways, self.runway_color)

        # Draw equally spaced markers
        left = camera_pos[0] - screen.get_width() / 2 - 200
        right = camera_pos[0] + screen.get_width() / 2
        for x in range(int(left), int(right)):
            if x % 300 != 0:
                continue
            screen_x = world_to_screen(np.array([x, 0.0]), camera_pos, screen.get_size())[0]
            line_surface = pg.Surface((1, screen.get_height() - y_ground), pg.SRCALPHA)
            line_surface.fill((255, 255, 255, 50))
            screen.blit(line_surface, (screen_x, y_ground))
            text = pg.font.Font(None, 24).render(str(x), True, (0, 0, 0))
            screen.blit(text, (screen_x + 5, y_ground + 5))
