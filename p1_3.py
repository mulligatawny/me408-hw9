import numpy as np
import matplotlib.pyplot as plt

N = np.array([64, 128, 256])

reg = np.array([1e-4, 2e-5, 4e-6])
rog = np.array([1e-3, 7e-4, 3e-4])

plt.semilogy(N, reg, 'ko-', label='regular')
plt.semilogy(N, rog, 'ro-', label='Rogallo')
plt.xlabel('N')
plt.ylabel('Largest stable $\Delta t$')
plt.grid(which='both')
plt.legend()
plt.show()
