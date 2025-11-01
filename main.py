import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def nonlinear_MRO(t, Y, m, gamma, k, F_interp, noise_interp, dim):

    x = Y[:dim]
    dxdt = Y[dim:]
    
    nl_term = 0.2 * np.sin(x) + 0.1 * (np.roll(x, 1) - x)

    F = F_interp(t)
    noise = noise_interp(t)

    dxdtt = -(gamma/m)*dxdt - (k/m)*x + nl_term + (F + noise)/m
    
    return np.concatenate([dxdt, dxdtt])

m = 1.0
gamma = 0.15
k = 1.0
dim = 3

t_start, t_end, t_points = 0.0, 60.0, 6000
t_eval = np.linspace(t_start, t_end, t_points)

def F_ext(t):
    return np.array([
        2.0 * np.exp(-0.5 * (t - 8)**2),
        np.sin(0.3 * t),
        0.5 * np.cos(0.6 * t)
    ])

noise_level = 0.05
rng = np.random.default_rng(42)
noise_data = noise_level * rng.normal(size=(dim, t_points))

F_data = np.array([F_ext(ti) for ti in t_eval])
F_interp = lambda t: np.array([interp1d(t_eval, F_data[:, i], kind='cubic', fill_value="extrapolate")(t) for i in range(dim)])
noise_interp = lambda t: np.array([interp1d(t_eval, noise_data[i], kind='linear', fill_value="extrapolate")(t) for i in range(dim)])

Y0 = np.concatenate([np.ones(dim)*0.5, np.zeros(dim)])

sol = solve_ivp(
    nonlinear_MRO,
    [t_start, t_end],
    Y0,
    args=(m, gamma, k, F_interp, noise_interp, dim),
    t_eval=t_eval,
    method='RK45'
)

t = sol.t
x = sol.y[:dim]

plt.figure(figsize=(12, 5))
for i in range(dim):
    plt.plot(t, x[i] + i*2, label=f"x_{i}(t) offset:{i*2}")
plt.title("Multidimensional Nonlinear MRO with Coupling, External Force and Noise")
plt.xlabel("Time")
plt.ylabel("Amplitude / Position")
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(7, 6))
plt.plot(x[0], x[1], lw=0.8)
plt.title("Projected trajectory (x₀ vs x₁)")
plt.xlabel("x₀")
plt.ylabel("x₁")
plt.grid()
plt.show()
