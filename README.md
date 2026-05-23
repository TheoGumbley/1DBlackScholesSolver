# Black-Scholes PDE Solver

This code computes a numerical solution of the 1D Black-Scholes equation for pricing European call and put options, implemented in python. Two time integration schemes are provided: an explicit Runge-Kutta 4 (RK4) solver and an implicit Crank-Nicolson (CN) solver.

A full technical write-up of the mathematics and implementation is included in `Black_Scholes_PDE_Technical_Document.pdf`.

---

## Methods

**Spatial discretisation**
- Central difference scheme applied to the Black-Scholes PDE on a uniform S-grid
- Second-order accurate in space: O(ΔS²)

**Time integration**
| Scheme | Type | Stability | Time Steps (T=5, n=200) |
|---|---|---|---|
| Runge-Kutta 4 (RK4) | Explicit | CFL-limited | ~8805 |
| Crank-Nicolson (CN) | Implicit | Unconditionally stable | 1500 |

**Validation**
- Solutions compared against the Black-Scholes closed-form analytical solution
- Relative error can be computed at each time step across the full S-grid

**Greeks**
- Δ (Delta) and Γ (Gamma) computed via second-order central difference (`numpy.gradient`) over the S-grid
- Θ (Theta) computed by interpolating option value at target asset price S across time steps

---

## Dependencies

```
numpy
scipy
matplotlib
```

Install with:

```bash
pip install numpy scipy matplotlib
```

---

## Usage

All parameters are set in the `Main Script` section, line 256 onwards in `black_scholes.py`:

```python
sigma = 0.2       # Annualised volatility
r     = 0.03      # Risk-free rate
S     = 100       # Asset price to evaluate
T     = 5         # Time to expiry (years)
K     = 120       # Strike price
n     = 200       # S-grid points
m     = 1500      # Time steps
type  = "Call"    # "Call" or "Put"

RK4explicit = False
CNimplicit  = True
```

Run with:

```bash
python black_scholes.py
```

The script outputs:
- Option value at the specified asset price S
- Solution vs closed-form plot across the full S-grid
- Error/residual convergence plot
- 3D surface plot of option value over asset price and time
- Greeks Δ, Γ, and Θ (with Θ validated against the closed-form result)



## Author

Theo Gumbley — Aerospace & CFD Engineer, Southampton University


