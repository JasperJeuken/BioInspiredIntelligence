import numpy as np

from aircraft import Aircraft2D, AircraftConfig
from environment import Environment
from terrain import Terrain
from evaluate import evaluate_population


def main():

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
        max_vertical_landing_speed = 10.0,
        control_effectiveness_speed = 50.0
    )
    environment = Environment(
        air_density = 1.225,
        gravity = 9.81
    )

    # Initialize terrain
    oceans = [(2300, 5000)]
    runways = [(200, 2000), (5200, 7000)]
    terrain = Terrain(oceans, runways)
    
    # Evaluate population
    best_controller = evaluate_population(config, environment, generations=100, population_size=100)
    print("Best controller weights (first layer):")
    print(best_controller.w1)
    print("Best controller weights (second layer):")
    print(best_controller.w2)
    print("Best controller biases (first layer):")
    print(best_controller.b1)
    print("Best controller biases (second layer):")
    print(best_controller.b2)


if __name__ == '__main__':
    main()
