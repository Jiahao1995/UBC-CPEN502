import numpy as np

x = np.linspace(-np.pi, np.pi, 100)


def func(n):
    return np.sin(2 * n) - 2 * n + 2 ^ n


y = map(func, x)


print(x)
print(list(y))


