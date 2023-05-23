import numpy as np
even = np.loadtxt("data/mnist_train.csv", delimiter=",", skiprows=1)
even_arr = np.empty_like(even)
rows, cols = even_arr.shape
even_len = len(even)
print("\n")
for i, x in enumerate(even):
    if x[0] == 2 or x[0] == 4 or x[0] == 6 or x[0] == 8 or x[0] == 0:
        even_arr = np.concatenate((even_arr, [x]), axis=0)
    even_mod = i % 50
    if even_mod == 0:
        print(f"\t{i*100/even_len:.2f}% Done", end='\r')
print("hello, here is the shape BEFORE", np.shape(even_arr))
even_arr = np.delete(even_arr, [i for i, x in enumerate(even)], 0)
print("hello, here is the shape AFTER", np.shape(even_arr))
np.savetxt("data/mnist_train_evens.csv", even_arr, fmt='%.0f', delimiter=' ')
