###############################################################################
# 1D Non-linear Advection Equation with Dispersion Term: Fourier Spectral     #
# Solver with Runge-Kutta IV Time Integration                                 #
###############################################################################

import numpy as np
import matplotlib.pyplot as plt

def compute_f(ui, k):
    uk = np.fft.fftshift(np.fft.fft(ui))/N
    up = np.fft.ifft(np.fft.ifftshift(1j*k*uk))*N
    f = ui*up
    return f

def fun(t, uk):
    ui = np.fft.ifft(np.fft.ifftshift(uk))*N
    f = compute_f(ui, k)
    fk = np.fft.fftshift(np.fft.fft(f))/N
    return -3*fk + 1j*(k**3)/4*uk

N = 128
L = 6
x = np.linspace(-3, 3, N+1)[:-1]
k = np.arange(-N/2, N/2)*2*np.pi/L
# initial condition
u0 = (64*np.exp(-12-8*x))/(1+np.exp(-12-8*x))**2 +\
     (4*np.pi**2*np.exp(np.pi-2*np.pi*x))/(1+np.exp(np.pi-2*np.pi*x))**2
u = u0
uk = np.fft.fftshift(np.fft.fft(u))/N
t = 0.0
tf = 0.1
dt = 0.00002
ukn = np.zeros_like(u)

while t < tf:
    k1 = dt*fun(t, uk)
    k2 = dt*fun(t+dt/2, uk+k1/2)
    k3 = dt*fun(t+dt/2, uk+k2/2)
    k4 = dt*fun(t+dt, uk+k3)
    ukn = uk + k1/6 + k2/3 + k3/3 + k4/6
    uk = ukn
    t = t + dt

sol = np.fft.ifft(np.fft.ifftshift(uk))*N
plt.title('Solution at t = {:.2f}'.format(t))
plt.plot(x, np.real(sol), 'o-', color='hotpink', label='N = {}'.format(N))
plt.plot(x, u0, 'k-',label='I.C.')
plt.xlabel('$x$')
plt.ylabel('$u$')
plt.legend()
plt.show()
