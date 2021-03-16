import numpy as np
import matplotlib.pyplot as plt

def fun(t, uk):
    return Sk 

N = 16
nu = 100
x = np.linspace(-1, 1, N+1)[:-1]
y = x
L = 2
k1 = np.arange(-N/2, N/2)*2*np.pi/L
k1 = k1**2
k2 = k1
kout = np.outer(k1, k2)

# meshing
X, Y = np.meshgrid(x,y)
# source 
S = 100*np.exp(-(X**2-1)**6 -(Y**2-1)**6) -60
Sk = np.fft.fftshift(np.fft.fft2(S))/N**2
# I.C.
u0 = (1 -X**2)**4*(1 -Y**2)**4

t = 0.0
tf = 0.01
dt = 0.000001

u = u0
uk = np.fft.fftshift(np.fft.fft2(u))/N**2
ukn = np.zeros_like(u)
rog = np.exp(-nu*(kout)*dt) # ROGALLO

while t < tf:
    k1 = dt*fun(t, uk)
    k2 = dt*fun(t+dt/2, uk+k1/2)
    k3 = dt*fun(t+dt/2, uk+k2/2)
    k4 = dt*fun(t+dt, uk+k3)
    ukn = (uk + k1/6 + k2/3 + k3/3 + k4/6)*rog
    uk = ukn
    t = t + dt

sol = np.fft.ifft2(np.fft.ifftshift(uk))*N**2
f1 = plt.figure(1)
plt.plot(x, np.real(sol[:,0]),'ro-')
plt.title('Solution at x = 0')
plt.show()

f2 = plt.figure(2)
plt.contourf(X, Y, np.real(sol))
plt.colorbar()
plt.show()

xy = np.zeros_like(x)
for i in range(N):
    xy[i] = sol[i,i]

f3 = plt.figure(3)
plt.plot(x, np.real(xy), 'b.-')
plt.title('Solution at x = y')
plt.show()
