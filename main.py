import pygame as pg
import numpy as np

from aircraft import Aircraft2D, AircraftConfig
from environment import Environment
from terrain import Terrain


def main():
    # Initialize PyGame
    pg.init()
    screen = pg.display.set_mode((1200, 800), pg.RESIZABLE)
    clock = pg.time.Clock()
    camera_pos = np.array([0.0, 150.0])
    font = pg.font.Font(None, 24)
    pg.display.set_caption('Aircraft simulation')

    # Initialize aircraft
    config = AircraftConfig(
        mass = 1000.0,
        max_thrust = 5000.0,
        reference_area = 10.0,
        lift_curve_slope = 5.0,
        parasite_drag_coefficient = 0.02,
        induced_drag_factor = 0.05,
        pitch_rate_gain = 2.0,
        max_control_surface_angle = np.radians(15.0),
        wheel_drag_coefficient = 0.1,
        stall_angle = np.radians(15.0),
        max_vertical_landing_speed = 10.0
    )
    environment = Environment(
        air_density = 1.225,
        gravity = 9.81
    )
    aircraft = [Aircraft2D(config, environment) for _ in range(50)]

    # Initialize terrain
    oceans = [(50, 100), (200, 300), (400, 450)]
    terrain = Terrain(oceans)

    # Main loop
    running = True
    while running:

        screen.fill((135, 206, 235))
        dt = clock.tick(60) / 1000
        fps = clock.get_fps()

        # Update aircraft state
        for ac in aircraft:
            ac.step(dt)
        max_x = max(ac.pos[0] for ac in aircraft)
        camera_pos = np.array([max_x, camera_pos[1]])

        # Draw FPS and max X position
        text = font.render(f'FPS: {fps:.2f}', True, (0, 0, 0))
        screen.blit(text, (10, 10))
        text = font.render(f'No. of aircraft: {len(aircraft)}', True, (0, 0, 0))
        screen.blit(text, (10, 30))
        text = font.render(f'Best X: {max_x:.2f}', True, (0, 0, 0))
        screen.blit(text, (10, 50))

        # Draw terrain
        terrain.draw(screen, camera_pos)

        # Draw aircraft
        for ac in aircraft:
            ac.draw(screen, camera_pos)

        # Handle events
        pg.display.flip()
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            if event.type == pg.VIDEORESIZE:
                screen = pg.display.set_mode((event.w, event.h), pg.RESIZABLE)

    pg.quit()


if __name__ == '__main__':
    main()
