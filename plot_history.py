import numpy as np
import matplotlib
import matplotlib.pyplot as plt


matplotlib.rc('font', size=18)


FILE = 'out\\20250826-212840\\replay.npz'


def main():
    data = np.load(FILE)
    pos_history = data['pos']
    vel_history = data['vel']
    pitch_history = data['pitch']
    thrust_history = data['thrust']
    control_surface_history = data['control_surface']
    brake_history = data['brake']
    time_history = data['time']

    plt.figure(figsize=(12, 10))

    plt.subplot(3, 2, 1)
    plt.plot(time_history, pos_history[:, 0])
    plt.xlabel('Time (s)')
    plt.ylabel('Position X (m)')
    plt.grid()

    plt.subplot(3, 2, 2)
    plt.plot(time_history, pos_history[:, 1])
    plt.xlabel('Time (s)')
    plt.ylabel('Position Y (m)')
    plt.grid()

    plt.subplot(3, 2, 3)
    plt.plot(time_history, vel_history[:, 0])
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity X (m/s)')
    plt.grid()

    plt.subplot(3, 2, 4)
    plt.plot(time_history, vel_history[:, 1])
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity Y (m/s)')
    plt.grid()

    plt.subplot(3, 2, 5)
    plt.plot(time_history, np.degrees(pitch_history))
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch Angle (degrees)')
    plt.grid()

    plt.subplot(3, 2, 6)
    plt.plot(time_history, thrust_history, label='Thrust')
    plt.plot(time_history, control_surface_history, label='Control Surface')
    plt.plot(time_history, brake_history, label='Brake')
    plt.xlabel('Time (s)')
    plt.ylabel('Control Inputs')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()