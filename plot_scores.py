import numpy as np
import matplotlib
import matplotlib.pyplot as plt


matplotlib.rc('font', size=18)


FILE = 'out\\20250824-184358\\generation_scores.npz'


def main():
    scores = np.load(FILE)['arr_0']
    print(len(scores))

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(np.arange(1, len(scores) + 1), scores)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best score')
    ax.grid()
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
