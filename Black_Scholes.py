# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as spy
from scipy.linalg import solve_banded
import time

# Define Fucntions ############################################################

def ComputeClosedFormCall(S, K, r, sigma, T): #closed form for validation
    d1 = (np.log(S/K) + (r+0.5*sigma**2)*T)/(sigma*T**0.5)
    d2 = d1 - sigma*T**0.5
    return S * spy.norm.cdf(d1) - K*np.exp(-r*T)*spy.norm.cdf(d2)

def ComputeClosedFormPut(S, K, r, sigma, T): #closed form for validation
    d1 = (np.log(S/K) + (r+0.5*sigma**2)*T)/(sigma*T**0.5)
    d2 = d1 - sigma*T**0.5
    return K*np.exp(-r*T)*spy.norm.cdf(-d2) - S*spy.norm.cdf(-d1)

def ComputeClosedFormThetaPut(S, K, r, sigma, tau): #true for a put
    if tau > 0:   #avoids dividing by zero at tau=0 (expiry)
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*tau) / (sigma*tau**0.5)
        d2 = d1 - sigma*tau**0.5
        theta = (-S * spy.norm.pdf(d1) * sigma / (2*tau**0.5) + r * K * np.exp(-r*tau) * spy.norm.cdf(-d2))
        return theta
    else:
        return np.nan

def ComputeClosedFormThetaCall(S, K, r, sigma, tau): #true for a put
    if tau > 0:   #avoids dividing by zero at tau=0 (expiry)
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*tau) / (sigma*tau**0.5)
        d2 = d1 - sigma*tau**0.5
        theta = (-S * spy.norm.pdf(d1) * sigma / (2*tau**0.5) - r * K * np.exp(-r*tau) * spy.norm.cdf(d2))
        return theta
    else:
        return np.nan
    
def V_i_call(S, K): #euro call payoff function
    return max(S-K, 0)

def V_i_put(S, K): #euro put payoff function
    return max(K-S, 0)
    

def f_i(V, S_grid, i, r, sigma, delta_s):
    "i=spacial loc, t = temporal loc"
    a_i = 0.5*sigma**2*S_grid[i]**2/delta_s**2 - r* S_grid[i]/(2*delta_s)
    b_i = -sigma**2 * S_grid[i]**2/delta_s**2 - r
    c_i = 0.5*sigma**2 * S_grid[i]**2 / delta_s**2 + r*S_grid[i]/(2*delta_s)
    
    return -(a_i*V[i-1] + b_i*V[i] + c_i*V[i+1])
    
def Compute_dV_dt(V, S_grid, r, sigma, delta_s):
    newV = np.zeros(len(V))
    for i in range(1, len(V)-1):
        newV[i] = f_i(V, S_grid, i, r, sigma, delta_s)     
    return newV

def RK4(V, S_grid, r, sigma, delta_s, delta_t, m, K, T, S_max, S, output_period, type, error):
    L2NormResidual,  V_stored, t_values, Iterations, OptionValues = np.array([]), np.array([]), np.array([]),  np.array([]),  np.array([])
    start = time.time()
    out_val, a = 0, 0
    
    for n in range(m-1, -1, -1):
        print(f"RK4 Solver Iteration: {abs(n-m)}")
        t_n = n * delta_t
        
        k1 = -delta_t*Compute_dV_dt(V, S_grid, r, sigma, delta_s)
        k2 = -delta_t*Compute_dV_dt(V +k1/2, S_grid, r, sigma, delta_s)
        k3 = -delta_t*Compute_dV_dt(V +k2/2, S_grid, r, sigma, delta_s)
        k4 =  -delta_t*Compute_dV_dt(V +k3, S_grid, r, sigma, delta_s)
        
        V_new = V + (k1 + 2*k2 + 2*k3 + k4)/6 #new V grid
        
        if type == "Call":
            V_new[-1] = S_max - K*np.exp(-r*(T-t_n)) #enforce BC's
            V_new[0] = 0 #enforce BC's
        if type == "Put":
            V_new[-1] = 0
            V_new[0] = K*np.exp(-r*(T-t_n))
       
        if error == True:
            tau = T - t_n
            if type=="Call":
                V_cf = np.array([ComputeClosedFormCall(s, K, r, sigma, tau) for s in S_grid])
            elif type=="Put":
                V_cf = np.array([ComputeClosedFormPut(s, K, r, sigma, tau) for s in S_grid])
            
            errors = [abs(V_new[i] - V_cf[i])/(abs(V_cf[i])+1e-10) for i in range(len(S_grid))]
            mean_error = sum(errors) / len(errors)
            L2NormResidual = np.append(L2NormResidual, mean_error)
            Iterations = np.append(Iterations, tau)
            
        elif error == False:
            Vsqrd = (V_new-V)**2 
            L2NormResidual = np.append(L2NormResidual, np.sqrt(delta_s*sum(Vsqrd)))
            Iterations = np.append(Iterations, abs(n-m))
        
        if a == out_val:
            if V_stored.size == 0:
                V_stored = V_new[np.newaxis, :]    
            else:
                V_stored = np.vstack([V_stored, V_new])
        
            t_values = np.append(t_values, T-t_n)
            out_val += output_period

        a+=1
        V=V_new
        OptionValue = FindOptionValue(S, S_grid, V)
        OptionValues = np.append(OptionValues, OptionValue)
    
    elapsed = time.time() - start
    return V, L2NormResidual, Iterations, elapsed, V_stored, t_values, OptionValues

def Crank_Nicholson(V, S_grid, r, sigma, delta_s, delta_t, m, K, T, S_max, S,output_period, type, error):
    L2NormResidual, Iterations, OptionValues, V_stored, t_values = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    start = time.time()
    out_val, a = 0, 0
    N = len(S_grid)
    
    a_i, b_i, c_i, alpha_i, beta_i, gamma_i = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    for i in range(1, N-1):
        a_i[i]     =  0.5*sigma**2*S_grid[i]**2/delta_s**2 - r*S_grid[i]/(2*delta_s)
        b_i[i]     = -sigma**2*S_grid[i]**2/delta_s**2 - r
        c_i[i]     =  0.5*sigma**2*S_grid[i]**2/delta_s**2 + r*S_grid[i]/(2*delta_s)
        alpha_i[i] = -a_i[i] * delta_t/2
        beta_i[i]  =  1 - delta_t/2 * b_i[i]
        gamma_i[i] = -c_i[i] * delta_t/2

    tilda_alpha_i = -alpha_i
    tilda_beta_i  =  2 - beta_i
    tilda_gamma_i = -gamma_i

    ab = np.zeros((3, N-2))
    
    ab[0, 1:]  = gamma_i[1:N-2]
    ab[1, :]   = beta_i[1:N-1]    
    ab[2, :-1] = alpha_i[2:N-1]
    

    for n in range(m-1, -1, -1):
        print(f"Crank-Nicholson Solver Iteration: {abs(n-m)}")
        t_n   = n * delta_t
        t_np1 = (n+1) * delta_t

        if type=="Call":
            V0_n   = 0
            V0_np1 = 0
            VN_n   = S_max - K * np.exp(-r * (T - t_n))
            VN_np1 = S_max - K * np.exp(-r * (T - t_np1))
            
        elif type=="Put":
            VN_n   = 0
            VN_np1 = 0
            V0_n   = K * np.exp(-r * (T - t_n))
            V0_np1 = K * np.exp(-r * (T - t_np1))

        V_rhs       = V.copy()
        V_rhs[0]    = V0_n
        V_rhs[-1]   = VN_n

        q = (tilda_alpha_i[1:N-1] * V_rhs[0:N-2]
           + tilda_beta_i[1:N-1]  * V_rhs[1:N-1]
           + tilda_gamma_i[1:N-1] * V_rhs[2:N])

        q[0]  -= alpha_i[1]   * V0_np1 #enforce d vector 
        q[-1] -= gamma_i[N-2] * VN_np1


        x = solve_banded((1, 1), ab, q)

        V_new        = np.zeros(N)
        V_new[1:-1]  = x
        V_new[0]     = V0_np1
        V_new[-1]    = VN_np1

        if error == True:
            tau = T - t_n
            if type=="Call":
                V_cf = np.array([ComputeClosedFormCall(s, K, r, sigma, tau) for s in S_grid])
            elif type=="Put":
                V_cf = np.array([ComputeClosedFormPut(s, K, r, sigma, tau) for s in S_grid])
            
            errors = [abs(V_new[i] - V_cf[i])/(abs(V_cf[i])+1e-10) for i in range(len(S_grid))]
            mean_error = sum(errors) / len(errors)
            
            L2NormResidual = np.append(L2NormResidual, mean_error)
            
            Iterations = np.append(Iterations, tau)
        
        elif error == False:
            Vsqrd = (V_new - V)**2
            L2NormResidual = np.append(L2NormResidual, np.sqrt(delta_s * sum(Vsqrd)))
            Iterations = np.append(Iterations, abs(n-m))
      
        if a == out_val:
            if V_stored.size == 0:
                V_stored = V_new[np.newaxis, :]      
            else:
                V_stored = np.vstack([V_stored, V_new])
        
            t_values = np.append(t_values, T-t_n)
            out_val += output_period
        
        a+=1
        V = V_new
        OptionValue = FindOptionValue(S, S_grid, V)
        OptionValues = np.append(OptionValues, OptionValue)

    elapsed = time.time() - start
    return V, L2NormResidual, Iterations, elapsed, V_stored, t_values, OptionValues

def FindOptionValue(S, Svals, Vvals):
    V = np.interp(S, Svals, Vvals)
    return V

def ExplictStability(delta_t, delta_s, sigma, S_max, m, T):
    """
    there is something worng with this function don't use- gives unstable time step'
    """
    while delta_t > 0.9 * (delta_s**2)/(sigma**2 * S_max**2):
        m += 5
        delta_t = T/m
    
    return m

def ComputeGreeks(V, S_grid,T_grid, OptionValues, S):
    delta = np.gradient(V, S_grid)
    gamma = np.gradient(delta, S_grid)
    delta_s = np.interp(S, S_grid, delta)
    gamma_s = np.interp(S, S_grid, gamma)
    theta_s = np.gradient(OptionValues, T_grid)
    return delta_s, gamma_s, theta_s





# Main Script ##################################################################
    
sigma = 0.2            #annualised volatility
r = 0.03               #risk free rate 
S = 100                # starting asset price
T = 5                  #expiry time (years)
K = 120                #strike price
n = 200                #S grid points (spatial discretisation)
m = 1500               #time steps #1000 for Rk4
output_period = 25     #period to output solution
RK4explicit = False      
CNimplicit = True
type = "Call"          #"Call" or "Put"
error = False          #for speed set error=False - gives a solver residual instead

S_max = 3*K                        #define max S-grid value
delta_s = S_max / (n-1)            #define grid spacing
S_grid  = np.linspace(0,S_max,n)   #create spatial grid
delta_t = T/m                      #define timestep


if type == "Call":
    V = np.array([V_i_call(i,K) for i in S_grid])
    ClosedForm = [ComputeClosedFormCall(i, K, r, sigma, T) for i in S_grid]
elif type == "Put":
    V = np.array([V_i_put(i,K) for i in S_grid])
    ClosedForm = [ComputeClosedFormPut(i, K, r, sigma, T) for i in S_grid]


if RK4explicit == True:
    while delta_t > 0.9 * (delta_s**2)/(sigma**2 * S_max**2):
        m += 5
        delta_t = T/m
    m_stable = m
    V, Residuals, Iterations, elapsed, V_stored, t_values, OptionValues = RK4(V, S_grid, r, sigma, delta_s, delta_t, m_stable, K, T, S_max, S, output_period, type, error)
    
elif CNimplicit == True:
    m_stable = m
    V, Residuals, Iterations, elapsed, V_stored, t_values, OptionValues = Crank_Nicholson(V, S_grid, r, sigma, delta_s, delta_t, m_stable, K, T, S_max, S, output_period, type, error)
    

plt.figure(figsize=(15,8))

ax1 = plt.subplot(1,2,1)
ax1.plot(S_grid, V, 'k-', label='Computed')  
ax1.plot(S_grid, ClosedForm, 'r--', label = 'Closed Form')
ax1.set_xlabel("Asset Price ($)", fontsize=16)  
ax1.set_ylabel(f"Euro {type} Option Value ($)", fontsize=16)
ax1.legend()
ax1.grid()

ax2 = plt.subplot(1,2,2)
ax2.plot(Iterations, Residuals, 'b-')
ax2.set_xlabel("Time to Expiry (yrs)", fontsize=16)
if error == True:
    ax2.set_ylabel("$\\epsilon$, Error", fontsize=16)
elif error == False:
    ax2.set_ylabel("L2 Norm Residual", fontsize=16)
ax2.grid()

plt.semilogy()
plt.show()


Value = FindOptionValue(S, S_grid, V)
print(f"\n{type} Option Value for Asset Price of ${S} is ${np.round(Value,2)}")
print(f"\nStable Time Step [yrs]: {np.round(T/m_stable, 7)}")
print(f"Wall Clock Time [s]: {elapsed:.2f}")


fig = plt.figure(figsize=(20, 7))

ax = fig.add_subplot(111, projection='3d')
T_mesh, S_mesh = np.meshgrid(t_values, S_grid)
V_mesh = V_stored.T   # shape must match the mesh
ax.plot_surface(T_mesh, S_mesh, V_mesh,
                cmap='viridis',
                edgecolor='none')

ax.view_init(elev=30, azim=-30)
ax.set_position([0.0, 0.0, 0.75, 1.0])
ax.set_xlabel("Time to Expiry (yrs)")
ax.set_ylabel("Asset Price, S ($)")
ax.set_title("Black–Scholes PDE Surface")

fig.text(0.57, 0.45, f'{type} Option Value, V ($)', 
         fontsize=10,
         rotation=87.5, 
         va='center', 
         ha='center')

plt.show()


T_grid = np.arange(0, T, delta_t)[::-1]
delta,gamma,theta = ComputeGreeks(V, S_grid, T_grid, OptionValues, S)
if type == "Call":
    ClosedTheta = [ComputeClosedFormThetaCall(S, K, r, sigma, i) for i in T_grid[::-1]]
elif type == "Put":
    ClosedTheta = [ComputeClosedFormThetaPut(S, K, r, sigma, i) for i in T_grid[::-1]]
    

plt.figure(figsize=(15,8))

ax1 = plt.subplot(1,2,1)
ax1.plot(T_grid, OptionValues, 'k-', label='computed')  
ax1.set_xlabel("Time to Expiry (yrs)", fontsize=16)  
ax1.set_ylabel(f"Euro {type} Option Value for Asset Price, S ($)", fontsize=16)
ax1.legend()
ax1.grid()

ax2 = plt.subplot(1,2,2)
ax2.plot(T_grid, theta, 'k-', label='Computed')
ax2.plot(T_grid, ClosedTheta,'r--', label='Closed Form')
ax2.set_ylabel("$\\Theta$", fontsize=16)
ax2.set_xlabel("Time to Expiry (yrs)", fontsize=16)
ax2.legend()
ax2.grid()

plt.show()


print(f"\n \u0394 = {delta:.4f}")
print("(Delta = how much option price changes for a move in the underlying asset price)")
print(f"\n \u0393 = {gamma:.6f}")
print("(Gamma = rate of change of delta, sensitivity of delta to the underlying asset price)")