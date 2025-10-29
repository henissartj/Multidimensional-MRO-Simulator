import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def nonlinear_MRO(t, Y, m, gamma, k, F_ext, noise_level, dim):
    # Y contains x_0, x_1,.. x_(dim-1), dx_0/dt,..dx_(dim-1)/dt
    x = Y[:dim]
    dxdt = Y[dim:]
    
    # Nonlinear term and coupling between dimensions
    nl_term = np.sin(x) * 0.2   # example of nonlinear coupling
    
    # External force and noise
    F = F_ext(t) + np.random.normal(0, noise_level, size=dim)
    
    # Multidimensional MRO equation with nonlinear coupling and force
    dxdtt = -(gamma/m)*dxdt - (k/m)*x + nl_term + F/m
    
    return np.concatenate([dxdt, dxdtt])

# Model parameters
m = 1.0
gamma = 0.15
k = 1.0
dim = 3 # system with 3 coupled dimensions, can be increased

# Definition of external force (a pulse or periodic signal)
def F_ext(t):
    # Example: simple pulse at t=8, periodic afterwards
    return np.array([2.0*np.exp(-0.5*(t-8)**2), np.sin(0.3*t), 0.5*np.cos(0.6*t)])

noise_level = 0.05 # modifiable noise

# Multidimensional initial conditions: x_i(0)=1.0 or 0.5 + v_i(0)=0
Y0 = np.concatenate([np.ones(dim)*0.5, np.zeros(dim)])

# Simulation time
t_start = 0.0
t_end = 60.0
t_points = 6000
t_eval = np.linspace(t_start, t_end, t_points)

# Numerical resolution with Runge-Kutta
sol = solve_ivp(nonlinear_MRO, [t_start, t_end], Y0, args=(m, gamma, k, F_ext, noise_level, dim),
                t_eval=t_eval, method='RK45')

# Results extraction
t = sol.t
x = sol.y[:dim] # x[0], x[1], x[2]...

# Visualization of trajectories in phase space
fig = plt.figure(figsize=(12, 5))

for i in range(dim):
    plt.plot(t, x[i] + i*2, label=f"x_{i}(t) offset:{i*2}") # vertical offset for readability

plt.title('Multidimensional MRO: oscillations, nonlinearity, external force and noise')
plt.xlabel('Time')
plt.ylabel('Amplitude / Position')
plt.grid()
plt.legend()
plt.show()

# Phase space display (2D projection)
plt.figure(figsize=(7,6))
plt.plot(x[0], x[1], lw=0.6)
plt.title('MRO attractor or projected trajectory (x_0 vs x_1)')
plt.xlabel('x_0')
plt.ylabel('x_1')
plt.grid()
plt.show()
