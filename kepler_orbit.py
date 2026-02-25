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

dt =  [0.125, 0.100, 0.075, 0.01]
Nt = 10000

def accecleration(rx, ry):
    r = np.sqrt(rx**2 + ry**2)
    ax = -rx / r**3
    ay = -ry / r**3
    return ax, ay

def RK2(rx, ry, vx, vy, dt):
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

def velocity_verlet(rx, ry, vx, vy, dt):
    for i in range(Nt - 1):
        ax, ay = accecleration(rx[i], ry[i])
        rx[i+1] = rx[i] + vx[i] * dt + 0.5 * ax * dt**2
        ry[i+1] = ry[i] + vy[i] * dt + 0.5 * ay * dt**2

        ax_next, ay_next = accecleration(rx[i+1], ry[i+1])
        vx[i+1] = vx[i] + 0.5 * (ax + ax_next) * dt
        vy[i+1] = vy[i] + 0.5 * (ay + ay_next) * dt

def total_energy(rx, ry, vx, vy, Nt):
    E_total = np.zeros(Nt)
    for i in range(Nt):
        r = np.sqrt(rx[i]**2 + ry[i]**2)
        kinetic_energy = 0.5 * (vx[i]**2 + vy[i]**2)
        potential_energy = -1 / r
        E_total[i] = kinetic_energy + potential_energy
    return E_total    

#Implemeting the numerical solution using the 2nd order Runge-Kutta method
rx_RK = np.zeros(Nt)
ry_RK = np.zeros(Nt)
rx_RK[0] = rx0
ry_RK[0] = ry0
vx_RK = np.zeros(Nt)
vy_RK = np.zeros(Nt)
vx_RK[0] = vx0
vy_RK[0] = vy0




# Implemeting the numerical solution using the velocity verlet method
rx_vv = np.zeros(Nt)
ry_vv = np.zeros(Nt)
rx_vv[0] = rx0
ry_vv[0] = ry0 
vx_vv = np.zeros(Nt)
vy_vv = np.zeros(Nt)
vx_vv[0] = vx0
vy_vv[0] = vy0

for i in range(len(dt)):
    velocity_verlet(rx_vv, ry_vv, vx_vv, vy_vv, dt[i])
    RK2(rx_RK, ry_RK, vx_RK, vy_RK, dt[i])

    plt.plot(rx0, ry0, 'bo', label='Initial Position', markersize=8)  # plot the initial position

    #plotting the RK2 numerical solution
    plt.plot(rx_RK, ry_RK, color='red', label='RK2 Solution', linewidth=2)  # plot the numerical solution
    #plotting the velocity verlet numerical solution
    plt.plot(rx_vv, ry_vv, color='blue', label='Velocity Verlet Solution', linewidth=2)  # plot the numerical solution
    #Plotting the annalytic solution
    plt.plot(x, y, 'k--', label='Analytic Solution')  # plot the analytic solution
    plt.legend(loc='upper left')
    plt.xlabel('x')
    plt.ylabel('y') 

    plt.setp(plt.gca().get_xticklabels(), fontsize=12)
    plt.setp(plt.gca().get_yticklabels(), fontsize=12)
    plt.title('Kepler Orbit (analytic solution vs Numerical solution) dt = ' + str(dt[i]))
    plt.savefig('kepler_orbit_dt' + str(dt[i]) + '.png')
    plt.show()

    #Energy RK2
    E_RK = total_energy(rx_RK, ry_RK, vx_RK, vy_RK, Nt)
    #Energy velocity verlet
    E_vv = total_energy(rx_vv, ry_vv, vx_vv, vy_vv, Nt)

    #plottting the total energy vs time for both methods
    plt.plot(np.arange(Nt) * dt[i], E_RK, color='red', label='RK2 Total Energy', linewidth=2)
    plt.plot(np.arange(Nt) * dt[i], E_vv, color='blue', label='Velocity Verlet Total Energy', linewidth=2)
    plt.xlabel('Time') 
    plt.ylabel('Total Energy')
    plt.setp(plt.gca().get_xticklabels(), fontsize=12)
    plt.setp(plt.gca().get_yticklabels(), fontsize=12)
    plt.title('Total Energy vs Time for Kepler Orbit dt = ' + str(dt[i]))
    plt.legend(loc='upper left')
    plt.savefig('kepler_energy_dt' + str(dt[i]) + '.png')
    plt.show()