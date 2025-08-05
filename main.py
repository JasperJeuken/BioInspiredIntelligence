import pygame as pg

from aircraft import Aircraft2D, AircraftConfig
from environment import Environment


def main():
    # Initialize PyGame
    pg.init()
    screen = pg.display.set_mode((800, 600))
    clock = pg.time.Clock()
    pg.display.set_caption("Aircraft simulation")

    # Initialize aircraft
    config = AircraftConfig(
        mass = 1000.0,
        max_thrust = 20000.0,
        reference_area= 16.0,
        lift_curve_slope = 5.0,
        parasite_drag_coefficient = 0.02,
        induced_drag_factor = 0.05,
        pitch_rate_gain = 2.0,
        max_control_surface_angle = 0.5
    )
    environment = Environment(
        air_density = 1.225,
        gravity = 9.81
    )
    aircraft = Aircraft2D(config, environment)
    aircraft.step(0.1)

    running = True
    while running:

        dt = clock.tick(60) / 1000

        # Handle events
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
        
        # Handle inputs
        pressed_keys = pg.key.get_pressed()
        if pressed_keys[pg.K_UP]:
            aircraft.thrust_setting += 0.3 * dt
        if pressed_keys[pg.K_DOWN]:
            aircraft.thrust_setting -= 0.3 * dt
        if pressed_keys[pg.K_LEFT]:
            aircraft.control_surface_angle -= 0.1 * dt
        if pressed_keys[pg.K_RIGHT]:
            aircraft.control_surface_angle += 0.1 * dt

        # Draw aircraft
        aircraft.draw(screen)

        screen.fill((0, 0, 0))
        pg.display.flip()

    pg.quit()

if __name__ == "__main__":
    main()
