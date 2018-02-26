import numpy as np
import matplotlib.pyplot as plt
import pickle

# IF YOU WANNA PLOT GRAPH OF EXPERIMENT WITH MORE LABELS CHANGE THE COEFFICIENTS
file_name = 'monotonous4_7labels.pickle'

arr1_total = np.array([])
arr2_total = np.array([])
arr3_total = np.array([])
arr4_total = np.array([])

with open(file_name, 'rb') as f:
    for i in range(100):
        arr1 = pickle.load(f)
        arr2 = pickle.load(f)
        arr3 = pickle.load(f)
        arr4 = pickle.load(f)

        if i == 0:
            x_arr1 = np.arange(len(arr1)) * 0.11
            x_arr2 = np.arange(len(arr2)) * 1.1
            x_arr3 = np.arange(len(arr3)) * 0.11
            x_arr4 = np.arange(len(arr4)) * 1.1

            arr1_total = arr1
            arr2_total = arr2
            arr3_total = arr3
            arr4_total = arr4
        else:
            arr1_total += arr1
            arr2_total += arr2
            arr3_total += arr3
            arr4_total += arr4

        # plt.ylim((0, 250))
        # plt.xlim((0, 500))
        # plt.plot(x_arr1, arr1, '-r', x_arr2, arr2, '-b', x_arr3, arr3, '-m', x_arr4, arr4, '-c')
        # plt.show()

arr1_total = arr1_total / 100
arr2_total = arr2_total / 100
arr3_total = arr3_total / 100
arr4_total = arr4_total / 100

# plt.ylim((0, 250))
# plt.xlim((0, 500))
plt.plot(x_arr1, arr1_total, '-r', x_arr2, arr2_total, '-b', x_arr3, arr3_total, '-m', x_arr4, arr4_total, '-c')
plt.show()
