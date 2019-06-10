import numpy as np


def main():
    a = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    print(a)
    b = [i-1 for i in a]
    print(b)
    c = [np.array([0, 0]), np.array([1, 1]), np.array([2, 2]), np.array([3, 3]), np.array([4, 4])]
    print(c)

    # for i in range(15):
    #     print(i)
    print('%02d'%(2))


if __name__ == '__main__':
    main()