import os
import time
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pygame as pg
import numpy as np

from aircraft import Aircraft2D, AircraftConfig
from environment import Environment
from terrain import Terrain
from genetic import GeneticAlgorithm
from controller import Controller


OUT_FOLDER = os.path.join('out', time.strftime('%Y%m%d-%H%M%S'))
os.makedirs(OUT_FOLDER, exist_ok=True)


np.random.seed(1)


def main():
    # Initialize PyGame
    pg.init()
    screen = pg.display.set_mode((1200, 800), pg.RESIZABLE)
    clock = pg.time.Clock()
    camera_pos = np.array([0.0, 150.0])
    font = pg.font.Font(None, 24)
    pg.display.set_caption('Aircraft simulation')

    # Set terrain and environment parameters
    oceans = [(2000, 5000)]
    runways = [(-400, 1400), (5600, 7400)]
    terrain = Terrain(oceans, runways)
    environment = Environment(
        air_density = 1.225,
        gravity = 9.81
    )

    # Set aircraft parameters
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
        max_vertical_landing_speed = 10.0,
        control_effectiveness_speed = 50.0
    )

    # Create GA
    ga = GeneticAlgorithm(population_size=200, elite_fraction=0.1, mutation_rate=0.09)
    controllers = [Controller() for _ in range(ga.population_size)]
    episode_time = 30.0  # [s]

    def reset_aircraft() -> list[Aircraft2D]:
        return [Aircraft2D(config, environment, terrain) for _ in range(ga.population_size)]
    
    aircraft = reset_aircraft()
    time = 0.0

    # Main loop
    running = True
    while running:

        screen.fill((135, 206, 235))
        dt = clock.tick(60) / 1000
        time += dt
        fps = clock.get_fps()

        # Control aircraft using GA controllers
        for ac, ctrl in zip(aircraft, controllers):
            state = np.array([
                ac.pos[0],
                ac.pos[1],
                ac.vel[0],
                ac.vel[1],
                ac.pitch,
                ac.pitch_rate
            ])
            thrust_cmd, control_surface_cmd = ctrl.forward(state)
            ac.thrust_setting = thrust_cmd
            ac.control_surface_angle = control_surface_cmd
            ac.step(dt)

        # Update camera position (follow best aircraft)
        max_x = max(ac.pos[0] for ac in aircraft if not ac.crashed)
        camera_pos = np.array([max_x, camera_pos[1]])

        # Draw terrain
        terrain.draw(screen, camera_pos)

        # Draw aircraft
        for ac in aircraft:
            ac.draw(screen, camera_pos, font)

        # Draw FPS and max X position
        text = font.render(f'FPS: {fps:.0f}', True, (0, 0, 0))
        screen.blit(text, (10, 10))
        text = font.render(f'No. of aircraft: {len(aircraft)}', True, (0, 0, 0))
        screen.blit(text, (10, 30))
        text = font.render(f'Generation: {ga.generation}', True, (0, 0, 0))
        screen.blit(text, (10, 50))
        text = font.render(f'Best X: {max_x:.0f}', True, (0, 0, 0))
        screen.blit(text, (10, 70))
        text = font.render(f'Time: {time:.1f}/{episode_time:.1f} s', True, (0, 0, 0))
        screen.blit(text, (10, 90))

        # Handle events
        pg.display.flip()
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            if event.type == pg.VIDEORESIZE:
                screen = pg.display.set_mode((event.w, event.h), pg.RESIZABLE)

        # Check episode end
        if time >= episode_time or all(ac.crashed for ac in aircraft):
            # Calculate scores for each aircraft
            scores = ga.evaluate(aircraft, terrain)
            
            # Save best controller
            filename = f'best_gen{ga.generation}.npz'
            best_controller = controllers[np.argmax(scores)]
            best_controller.save(os.path.join(OUT_FOLDER, filename))
            print(f'Generation {ga.generation} best score: {max(scores):.2f}')

            # Create next generation
            controllers = ga.next_generation(controllers, scores)
            aircraft = reset_aircraft()
            time = 0.0
            episode_time += 1.0
            episode_time = min(episode_time, 90.0)

    pg.quit()


if __name__ == '__main__':
    main()
