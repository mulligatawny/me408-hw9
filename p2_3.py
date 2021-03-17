import numpy as np
import matplotlib.pyplot as plt

def fun(t, uk):
    return -nu*(kout*uk) + Sk 

N = 16
nu = 100
x = np.linspace(-1, 1, N+1)[:-1]
y = x
L = 2
n1 = np.arange(-N/2, N/2)*2*np.pi/L
n1 = n1**2
n2 = n1
kout = np.add.outer(n1, n2)

X, Y = np.meshgrid(x,y)
# source 
S = 100*np.exp(-(X**2-1)**6 -(Y**2-1)**6) -60
Sk = np.fft.fftshift(np.fft.fft2(S))/N**2
# I.C.
u0 = (1 -X**2)**4*(1 -Y**2)**4
from scipy.linalg import expm, sinm, cosm
def solve(tf):
    t = 0.0
    #tf = 0.01
    dt = 0.0001

    rog = np.exp(-nu*(kout)*dt) # ROGALLO
    rog2 = np.exp(-nu*(kout)*dt/2) # ROGALLO
    u = u0
    uk = np.fft.fftshift(np.fft.fft2(u))/N**2
    ukn = np.zeros_like(uk)

    while t < tf:
        k1 = dt*fun(t, uk)
        k2 = dt*fun(t+dt/2, uk+k1/2)
        k3 = dt*fun(t+dt/2, uk+k2/2)
        k4 = dt*fun(t+dt, uk+k3)
        ukn = (uk*rog + k1/6*rog + k2/3*rog2 + k3/3*rog2 + k4/6*rog)
        uk = ukn
        t = t + dt

    return np.fft.ifft2(np.fft.ifftshift(uk))*(N**2)

times = np.array([0.00025, 0.0005, 0.00075, 0.001, 0.01])
#times = np.array([0.01])
sols = np.zeros(((N, N, len(times))), dtype='complex')

fig1 = 0

for i in range(len(times)):
    sols[:,:,i] = solve(times[i])    
    print('Next ...')
    if fig1:
        plt.plot(y, np.real(sols[:,int(len(x)/2),i]),'o-', label='t = {:.5f}'.format(times[i]))
        plt.xlabel('$y$')
        plt.ylabel('$u$')
        plt.title('Solution at $x=0$')
        plt.grid()
        plt.legend()

    if not fig1:
        plt.plot(x, np.real(sols[int(len(x)/2),:,i]),'o-', label='t = {:.5f}'.format(times[i]))
        plt.xlabel('$x$')
        plt.ylabel('$u$')
        plt.title('Solution at $y=0$')
        plt.grid()
        plt.legend()

plt.show()

xy = np.zeros_like(x, dtype='complex')
plot_xy = 1
for i in range(len(times)):
    for j in range(N):
        xy[j] = sols[j,j,i]
    if plot_xy == 1:
        plt.plot(x, np.real(xy), 'o-',label='t = {:.5f}'.format(times[i]))
        plt.xlabel('Spatial coordinate')
        plt.ylabel('$u$')
        plt.title('Solution at $x=y$')
        plt.grid()
        plt.legend()


#plt.contourf(X, Y, np.real(sols[:,:,-1]))
#plt.colorbar()
#plt.show()
