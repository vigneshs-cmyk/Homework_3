import numpy as np
import matplotlib.pyplot as plt
import math




# analytic solution for Kepler's equation
theta = np.linspace(0, 2 * np.pi, 10000)
e = -0.91
p = 0.9
r = p / (1 + e * np.cos(theta))
x = r * np.cos(theta)
y = r * np.sin(theta)


#initial setup for the numerical solution
rx0 = 10
ry0 = 0
vx0 = 0
vy0 = 0.3/(np.sqrt(rx0))

dt =  0.01
Nt = 10000

def accecleration(rx, ry):
    r = np.sqrt(rx**2 + ry**2)
    ax = -rx / r**3
    ay = -ry / r**3
    return ax, ay


#Implemeting the numerical solution using the 2nd order Runge-Kutta method
rx = np.zeros(Nt)
ry = np.zeros(Nt)
rx[0] = rx0
ry[0] = ry0
vx = np.zeros(Nt)
vy = np.zeros(Nt)
vx[0] = vx0
vy[0] = vy0
ax = np.zeros(Nt)
ay = np.zeros(Nt)
ax[0], ay[0] = accecleration(rx[0], ry[0])


def RK2(rx, ry, vx, vy, ax, ay, dt):
    for i in range(Nt - 1):
        ax1, ay1 = accecleration(rx[i], ry[i])          # k1 for velocity
        k1x, k1y = vx[i], vy[i]
        k1vx, k1vy = ax1, ay1

        rx_mid = rx[i] + 0.5 * dt * k1x
        ry_mid = ry[i] + 0.5 * dt * k1y
        vx_mid = vx[i] + 0.5 * dt * k1vx
        vy_mid = vy[i] + 0.5 * dt * k1vy

        ax2, ay2 = accecleration(rx_mid, ry_mid)        # k2 for velocity

        rx[i+1] = rx[i] + dt * vx_mid
        ry[i+1] = ry[i] + dt * vy_mid
        vx[i+1] = vx[i] + dt * ax2
        vy[i+1] = vy[i] + dt * ay2

RK2(rx, ry, vx, vy, ax, ay, dt)



#plotting the RK2 numerical solution
plt.plot(rx[0], ry[0], 'bo', label='Initial Position')  # plot the initial position
plt.plot(rx, ry, color='orange', label='RK2 Solution')  # plot the numerical solution

#Plotting the annalytic solution
plt.plot(x, y, '--', label='Analytic Solution')  # plot the analytic solution
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Kepler Orbit (analyitc solution vs Numberical solution)')
plt.savefig('kepler_orbit.png')
plt.show()