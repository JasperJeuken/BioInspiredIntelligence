"""Camera utilities"""
import numpy as np


def world_to_screen(pos: np.ndarray, camera_pos: np.ndarray, screen_size: tuple[int, int]) -> np.ndarray:
    """Convert world coordinates to screen coordinates

    Args:
        pos (np.ndarray): world position
        camera_pos (np.ndarray): camera position
        screen_size (tuple[int, int]): screen size

    Returns:
        np.ndarray: screen position
    """
    screen_center = np.array(screen_size) / 2
    relative = pos - camera_pos
    relative[1] = -relative[1]  # invert y-axis
    return relative + screen_center


def screen_to_world(screen_pos: np.ndarray, camera_pos: np.ndarray, screen_size: tuple[int, int]) -> np.ndarray:
    """Convert screen coordinates to world coordinates

    Args:
        screen_pos (np.ndarray): screen position
        camera_pos (np.ndarray): camera position
        screen_size (tuple[int, int]): screen size

    Returns:
        np.ndarray: world position
    """
    screen_center = np.array(screen_size) / 2
    relative = screen_pos - screen_center
    relative[1] = -relative[1]  # invert y-axis
    return relative + camera_pos
