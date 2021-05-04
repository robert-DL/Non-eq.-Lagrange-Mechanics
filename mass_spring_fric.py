import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.optimize as opt

class MassSpringSchemes:
    def __init__(self, S_0, T_0, k, mu, h, m, R, N):
        # Parameters
        self.S_0 = S_0
        self.T_0 = T_0
        self.k = k
        self.mu = mu
        self.h = h
        self.m = m
        self.R = R
        self.N = N

        self.omega = k/m

    def simulate(self, t, h, X, S, scheme):
        for j in range(1, len(t)-1):
            if scheme == 1:
                X[j + 1], S[j + 1] = self.verlet_scheme(
                    h, X[j], X[j-1], S[j])
            elif scheme == 2:
                X[j + 1], S[j + 1] = self.midpoint_scheme(
                    h, X[j], X[j - 1], S[j], S[j-1])
            else:
                X[j + 1], S[j + 1] = self.sym_variational_scheme(
                    h, X[j], X[j - 1], S[j])

        return X, S

    def verlet_scheme(self, h, x_k, x_km, s_k):
        # Update position
        fric = 1/(1 + self.mu*h/self.m)
        coeff_k = self.mu*h/self.m - self.omega*pow(h, 2) + 2
        x_kp = fric*(coeff_k*x_k - x_km)

        # Update Entropy
        exp_k = (s_k - self.S_0)/(self.R*self.N)
        T_k = self.T_0*np.exp(exp_k)
        s_kp = (h*self.mu/T_k)*((x_kp - x_k)/h)**2 + s_k

        return x_kp, s_kp


    def midpoint_scheme(self, h, x_k, x_km, s_k, s_km):
        # Position
        coeff_inv = 1/(2 + self.omega*(h**2) + self.mu*h/self.m)
        coeff_k = 4 - 2*self.omega*(h**2)
        coeff_km = 2 - self.omega*(h**2) - self.mu*h/self.m

        x_kp = coeff_inv*(coeff_k*x_k - coeff_km*x_km)

        # Entropy (solve zero point problem)
        s_kp  = opt.fsolve(self.mp_root_eqn,
                           s_k, (s_k, s_km, x_kp, x_k, h))

        return x_kp, s_kp

    def mp_root_eqn(self, s_kp, s_k, s_km, x_kp, x_k, h):
        # Intermediate values
        coeff = 1/(self.N*self.R)
        exp_m = (s_km + s_k)/2 - self.S_0
        exp_p = (s_k + s_kp)/2 - self.S_0
        fwd_s = (s_kp - s_k)/h
        fwd_x = (x_kp - x_k)/h

        # Function to evaluate
        f = 0.5*(self.T_0*np.exp(coeff*exp_m)
                 + self.T_0*np.exp(coeff*exp_p))*fwd_s - self.mu*fwd_x**2

        return f

    def sym_variational_scheme(self, h, x_k, x_km, s_k):
        # Position update
        coeff_inv = 1/(1 + 0.5*h*(self.mu/self.m))
        coeff_k = (2 - self.omega*(h**2))
        coeff_km = 0.5*h*(self.mu/self.m) - 1

        x_kp = coeff_inv*(coeff_km*x_km + coeff_k*x_k)

        s_kp  = opt.fsolve(self.var_root_eqn,
                           s_k, (s_k, x_kp, x_k, h))

        return x_kp, s_kp

    def var_root_eqn(self, s_kp, s_k, x_kp, x_k, h):
        # Intermediate values
        coeff = 1/(self.R*self.N)
        exp_k = coeff*(s_k - self.S_0)
        exp_kp = coeff*(s_kp - self.S_0)

        fwd_s = (s_kp - s_k)/h
        fwd_x = (x_kp - x_k)/h

        # Function to evaluate
        f = 0.5*(self.T_0*np.exp(exp_k)
                 + self.T_0*np.exp(exp_kp))*fwd_s - self.mu*fwd_x**2

        return f


    def set_mu(self, mu):
        self.mu = mu


# Parameters
h = 1*10**(-4)
mu = [0, 0.2, 5, 10]
m = 5
N = 1
k = 5
R = 8.31446261815324
S_0 = 0
T_0 = 300
x_0 = 0.3

# Set up integrator
system = MassSpringSchemes(S_0, T_0, k, mu[0], h, m, R, N)

# Time, position, entropy initialized
t = np.arange(0, 100 + h, h)
X = np.zeros((len(mu), np.size(t)+1))
S = np.zeros((len(mu), np.size(t)))

X[:, 0] = x_0
X[:, 1] = x_0

# Plots for entropy and position
matplotlib.rcParams.update({'font.size': 16})

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
fig2, ax2 = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))

# Simulate
for i, mu_i in enumerate(mu):
    system.set_mu(mu_i)
    if i <= 1:
        row = 0
    else:
        row = 1
    X[i, :], S[i, :] = system.simulate(t, h, X[i, :], S[i, :], 3)


    ax[row, i%2].plot(t, X[i, 1:])
    ax[row, i%2].set_title(r"$\lambda = " + str(mu_i) + "$")
    ax[row, i%2].grid(True)
    ax[row, i%2].set_xticks(range(0, 110, 10))
    ax[row, i%2].set_xlabel(r"$t$")
    ax[row, i%2].set_ylabel(r"$x$")

    ax2[row, i%2].plot(t, S[i, :])
    ax2[row, i%2].set_title(r"$\lambda = " + str(mu_i) + "$")
    ax2[row, i%2].grid(True)
    ax2[row, i%2].set_xticks(range(0, 110, 10))
    ax2[row, i%2].set_xlabel(r"$t$")
    ax2[row, i%2].set_ylabel(r"$S$")

fig.tight_layout()
fig.show()
#fig.savefig("all_positions.pdf", dpi=300)

fig2.tight_layout()
fig2.show()
#fig2.savefig("all_entropy.pdf", dpi=300)




