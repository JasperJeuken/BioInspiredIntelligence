import numpy as np
import pygame as pg

from aircraft import Aircraft2D, AircraftConfig
from camera import world_to_screen
from environment import Environment


def main():
    # Initialize PyGame
    pg.init()
    screen = pg.display.set_mode((1200, 800))
    clock = pg.time.Clock()
    camera_pos = np.array([0.0, 0.0])
    font = pg.font.Font(None, 36)
    pg.display.set_caption("Aircraft simulation")

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
    aircraft = Aircraft2D(config, environment)

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
            aircraft.control_surface_angle += 0.1 * dt
        if pressed_keys[pg.K_RIGHT]:
            aircraft.control_surface_angle -= 0.1 * dt

        # Update aircraft state
        aircraft.step(dt)
        camera_pos = aircraft.pos.copy()

        # Draw aircraft state as text
        screen.fill((0, 0, 0))
        pos_text = font.render(f'Position: [{aircraft.pos[0]:.1f} {aircraft.pos[1]:.1f}]', True, (255, 255, 255))
        vel_text = font.render(f'Velocity: [{aircraft.vel[0]:.1f} {aircraft.vel[1]:.1f}]', True, (255, 255, 255))
        thrust_text = font.render(f'Thrust: {aircraft.thrust_setting:.2f}', True, (255, 255, 255))
        control_text = font.render(f'Control Surface: {aircraft.control_surface_angle:.2f} rad', True, (255, 255, 255))
        pitch_text = font.render(f'Pitch: {aircraft.pitch:.2f} rad', True, (255, 255, 255))
        lift_text = font.render(f'Lift: {aircraft.forces["lift"][1]:.2f} N', True, (255, 255, 255))
        drag_text = font.render(f'Drag: {aircraft.forces["drag"][0]:.2f} N', True, (255, 255, 255))
        gravity_text = font.render(f'Gravity: {aircraft.forces["gravity"][1]:.2f} N', True, (255, 255, 255))
        wheel_text = font.render(f'Wheel Drag: {aircraft.forces["wheel_drag"][0]:.2f} N', True, (255, 255, 255))
        stall_text = font.render(f'Stalled: {aircraft.stalled}', True, (255, 0, 0) if aircraft.stalled else (0, 255, 0))
        crash_text = font.render(f'Crashed: {aircraft.crashed}', True, (255, 0, 0) if aircraft.crashed else (0, 255, 0))
        screen.blit(pos_text, (10, 10))
        screen.blit(vel_text, (10, 40))
        screen.blit(thrust_text, (10, 70))
        screen.blit(control_text, (10, 100))
        screen.blit(lift_text, (10, 130))
        screen.blit(drag_text, (10, 160))
        screen.blit(gravity_text, (10, 190))
        screen.blit(wheel_text, (10, 220))
        screen.blit(stall_text, (10, 250))
        screen.blit(pitch_text, (10, 280))
        screen.blit(crash_text, (10, 310))

        # Draw ground
        y_ground = world_to_screen(np.array([0.0, 0.0]), camera_pos, screen.get_size())[1]
        pg.draw.line(screen, (50, 200, 50), (0, y_ground), (screen.get_width(), y_ground), 2)

        # Draw aircraft
        aircraft.draw(screen, camera_pos)
        pg.display.flip()

    pg.quit()

if __name__ == "__main__":
    main()
