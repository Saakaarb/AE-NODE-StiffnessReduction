## Reaction System Modeled

Two-step oxidation with an implicit inert bath gas at constant pressure:

1. **Fuel oxidation:**  
   $$
   \mathrm{F} + 1.5\ \mathrm{O_2} \ \xrightarrow{k_1}\ \mathrm{CO} + 2\ \mathrm{H_2O}
   $$
   $k_1 = A_1 e^{-E_1/(RT)},\quad A_1 = 1.0\times 10^6\ \mathrm{s^{-1}},\quad E_1 = 6.0\times 10^4\ \mathrm{J\,mol^{-1}}$

2. **CO oxidation:**  
   $$
   \mathrm{CO} + 0.5\ \mathrm{O_2} \ \xrightarrow{k_2}\ \mathrm{CO_2}
   $$
   $k_2 = A_2 e^{-E_2/(RT)},\quad A_2 = 5.0\times 10^5\ \mathrm{s^{-1}},\quad E_2 = 4.0\times 10^4\ \mathrm{J\,mol^{-1}}$

---

## Governing ODE System

State vector:
$$
y = \left[ Y_F,\ Y_{O2},\ Y_{CO},\ Y_{CO2},\ T \right]
$$

### Concentrations (mol m⁻³)
$$
C_F = \frac{\rho Y_F}{MW_F}, \quad 
C_{O2} = \frac{\rho Y_{O2}}{MW_{O2}}, \quad
C_{CO} = \frac{\rho Y_{CO}}{MW_{CO}}
$$

### Reaction rates
$$
\begin{aligned}
k_1 &= A_1 e^{-E_1/(RT)}, \quad 
k_2 = A_2 e^{-E_2/(RT)},\\
r_1 &= k_1 \, C_F \, C_{O2}, \quad
r_2 = k_2 \, C_{CO} \, C_{O2}^{0.5}
\end{aligned}
$$

### Species equations (mass fractions)
$$
\begin{aligned}
\dot{Y}_F     &= -\frac{MW_F}{\rho} \, r_1,\\
\dot{Y}_{O2}  &= -\frac{MW_{O2}}{\rho} \, (1.5\,r_1 + 0.5\,r_2),\\
\dot{Y}_{CO}  &= \frac{MW_{CO}}{\rho} \, (r_1 - r_2),\\
\dot{Y}_{CO2} &= \frac{MW_{CO2}}{\rho} \, r_2
\end{aligned}
$$

### Energy equation
$$
\dot{T} = \frac{Q_1 r_1 + Q_2 r_2}{\rho\, C_p^{\mathrm{mix}}}
$$
with:
- $Q_1 = 3.0\times 10^5\ \mathrm{J\,mol^{-1}}$
- $Q_2 = 2.0\times 10^5\ \mathrm{J\,mol^{-1}}$  
- $C_p^{\mathrm{mix}} = 1500\ \mathrm{J\,(kg\,K)^{-1}}$

---

## Initial Condition Ranges

For each generated case ($N=100$ by default):

| Variable       | Distribution / Value                |
|----------------|-------------------------------------|
| $Y_F$          | Uniform(0.03, 0.10)                  |
| $Y_{O2}$       | 0.21 (fixed)                         |
| $Y_{CO}$       | 0                                   |
| $Y_{CO2}$      | 0                                   |
| $T$ [K]        | Uniform(1100, 1300)                  |

A safeguard ensures $Y_F + Y_{O2} \le 1$ (not triggered in given ranges).

---

## Integration Setup

- Method: **Radau** (stiff, implicit)
- Time span: $0 \ \text{to}\ 5\ \mathrm{ms}$  
- Tolerances:  
  - rtol = $10^{-7}$  
  - atol = $10^{-10}$ (species)  
  - atol = $10^{-6}$ (temperature)  
