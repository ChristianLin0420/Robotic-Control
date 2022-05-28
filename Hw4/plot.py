import os
import numpy as np
from matplotlib import pyplot as plt

def main():
    save_dir = './save'
    its = []
    mean_returns = []
    std_returns = []

    with open(os.path.join(save_dir, 'return.txt')) as file:
        lines = file.read().split()

    for line in lines:
        strs = line.split(',')
        if len(strs) < 3: continue

        its.append(int(strs[0]))
        mean_returns.append(float(strs[1]))
        std_returns.append(float(strs[2]))

    its = np.array(its, dtype=np.int32)
    mean_returns = np.array(mean_returns, dtype=np.float32)
    std_returns = np.array(std_returns, dtype=np.float32)

    plt.xlim(0, its[-1])
    plt.ylim(0, 250)
    plt.ylabel('Total reward')
    plt.xlabel('Iteration')
    plt.grid()
    plt.plot(its, mean_returns, color='red')
    plt.fill_between(
        its,
        mean_returns+std_returns,
        mean_returns-std_returns,
        color='red',
        alpha=0.4
    )
    plt.show()


if __name__ == '__main__':
    main()