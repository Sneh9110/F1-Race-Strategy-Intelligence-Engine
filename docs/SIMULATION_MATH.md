# Race Simulation Engine - Mathematical Reference

## Overview

This document provides the mathematical foundations and formulas used in the Race Simulation Engine.

---

## 1. Lap Time Calculation

### Base Lap Time Formula

$$
T_{lap} = T_{base} + \Delta T_{tire} + \Delta T_{fuel} + \Delta T_{traffic} + \Delta T_{SC}
$$

Where:
- $T_{base}$ = Base lap time (track-specific)
- $\Delta T_{tire}$ = Tire degradation effect
- $\Delta T_{fuel}$ = Fuel load effect
- $\Delta T_{traffic}$ = Dirty air penalty
- $\Delta T_{SC}$ = Safety car effect

### Tire Degradation Effect

$$
\Delta T_{tire} = age_{tire} \times r_{deg}(compound)
$$

Degradation rates by compound:
- Soft: $r_{deg} = 0.05$ s/lap
- Medium: $r_{deg} = 0.03$ s/lap
- Hard: $r_{deg} = 0.02$ s/lap

**Example:**
- Tire age: 20 laps
- Compound: Soft
- $\Delta T_{tire} = 20 \times 0.05 = 1.0$ seconds

### Fuel Load Effect

$$
\Delta T_{fuel} = m_{fuel} \times k_{fuel}
$$

Where:
- $m_{fuel}$ = Current fuel load (kg)
- $k_{fuel}$ = Fuel effect coefficient (≈0.03 s/kg)

**Example:**
- Fuel load: 50 kg
- $\Delta T_{fuel} = 50 \times 0.03 = 1.5$ seconds

### Traffic Effect

$$
\Delta T_{traffic} = \begin{cases}
0.3 \text{ s} & \text{if } \Delta t_{ahead} < 1.0 \text{ s (dirty air)} \\
0 & \text{if } \Delta t_{ahead} \geq 1.0 \text{ s (clean air)}
\end{cases}
$$

Where $\Delta t_{ahead}$ is the gap to car ahead.

### Safety Car Effect

$$
\Delta T_{SC} = T_{lap} \times (m_{SC} - 1)
$$

Where $m_{SC} = 1.3$ (30% slower under SC)

### Complete Example

**Scenario:**
- Base lap time: 90.0 s
- Tire age: 15 laps (Medium compound)
- Fuel load: 80 kg
- Gap to ahead: 0.5 s (dirty air)
- No safety car

$$
\begin{align*}
T_{lap} &= 90.0 + (15 \times 0.03) + (80 \times 0.03) + 0.3 + 0 \\
&= 90.0 + 0.45 + 2.4 + 0.3 \\
&= 93.15 \text{ seconds}
\end{align*}
$$

---

## 2. Tire Degradation Model

### Degradation Rate Function

$$
D(t, c, T, L) = D_{base}(c) \times f_{temp}(T) \times f_{load}(L) \times (1 + \epsilon_t)
$$

Where:
- $c$ = Tire compound (Soft/Medium/Hard)
- $T$ = Track temperature
- $L$ = Fuel load
- $\epsilon_t \sim \mathcal{N}(0, 0.02)$ = Random noise (Monte Carlo)

### Base Degradation Rates

$$
D_{base}(c) = \begin{cases}
0.05 & \text{Soft} \\
0.03 & \text{Medium} \\
0.02 & \text{Hard}
\end{cases}
$$

### Temperature Effect

$$
f_{temp}(T) = 1 + k_T \times (T - T_{opt})
$$

Where:
- $k_T = 0.01$ per °C
- $T_{opt} = 35°C$ (optimal track temp)

**Example:**
- Track temp: 45°C
- $f_{temp}(45) = 1 + 0.01 \times (45 - 35) = 1.1$ (10% faster degradation)

### Fuel Load Effect

$$
f_{load}(L) = 1 + k_L \times \frac{L - L_{min}}{L_{max} - L_{min}}
$$

Where:
- $k_L = 0.15$ (15% max increase)
- $L_{min} = 0$ kg, $L_{max} = 110$ kg

---

## 3. Pit Stop Loss Model

### Total Pit Loss

$$
L_{pit} = t_{stationary} + t_{in} + t_{out} + L_{traffic}
$$

Where:
- $t_{stationary}$ = Time stopped in pit box (≈2.5 s)
- $t_{in}$ = Pit entry time loss
- $t_{out}$ = Pit exit time loss
- $L_{traffic}$ = Traffic-dependent loss

### Traffic-Dependent Loss

$$
L_{traffic} = k_{traffic} \times n_{cars}
$$

Where:
- $n_{cars}$ = Number of cars in pit window (±10 positions)
- $k_{traffic} = 0.5$ s per car

**Example:**
- Base pit loss: 22.0 s
- Cars in window: 3
- $L_{traffic} = 0.5 \times 3 = 1.5$ s
- Total: $22.0 + 1.5 = 23.5$ seconds

### Safety Car Discount

$$
L_{pit}^{SC} = L_{pit} \times 0.5
$$

Pit loss is halved under safety car conditions.

---

## 4. Safety Car Probability Model

### Base Probability

$$
P_{SC}(lap) = P_{base} \times f_{progress}(lap) \times f_{incidents}(lap)
$$

### Race Progress Factor

$$
f_{progress}(lap) = \begin{cases}
1.2 & \text{if } lap < 15 \text{ (early)} \\
1.0 & \text{if } 15 \leq lap \leq 50 \text{ (mid)} \\
0.8 & \text{if } lap > 50 \text{ (late)}
\end{cases}
$$

### Incident Factor

$$
f_{incidents}(lap) = 1 + 0.3 \times n_{incidents}^{recent}
$$

Where $n_{incidents}^{recent}$ = incidents in last 5 laps

---

## 5. Position Calculation

### Cumulative Time-Based Positions

$$
pos_i = \text{rank}\left(\sum_{l=1}^{L} T_{lap}^{i,l}\right)
$$

Where $T_{lap}^{i,l}$ is lap time for driver $i$ on lap $l$.

### Gap Calculation

$$
\Delta t_i = \sum_{l=1}^{L} T_{lap}^{i,l} - \sum_{l=1}^{L} T_{lap}^{leader,l}
$$

---

## 6. Monte Carlo Statistics

### Win Probability

$$
P_{win} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}(pos_i = 1)
$$

Where:
- $N$ = Number of MC runs
- $\mathbb{1}(\cdot)$ = Indicator function

### Position Distribution

$$
P(pos = k) = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}(pos_i = k)
$$

### Expected Race Time

$$
\mathbb{E}[T_{race}] = \frac{1}{N} \sum_{i=1}^{N} T_{race}^i
$$

### Variance

$$
\text{Var}(T_{race}) = \frac{1}{N-1} \sum_{i=1}^{N} (T_{race}^i - \mathbb{E}[T_{race}])^2
$$

### Convergence Criterion

MC simulation converges when:

$$
\text{Var}(P_{win}^{[t-w:t]}) < \epsilon
$$

Where:
- $w$ = Window size (100 runs)
- $\epsilon$ = Threshold (0.01)
- $P_{win}^{[t-w:t]}$ = Win probability in window

---

## 7. Noise Injection (Monte Carlo)

### Tire Degradation Noise

$$
D' = D \times (1 + \epsilon_{deg})
$$

Where $\epsilon_{deg} \sim \mathcal{N}(0, 0.02)$

### Lap Time Noise

$$
T_{lap}' = T_{lap} + \epsilon_{lap}
$$

Where $\epsilon_{lap} \sim \mathcal{N}(0, 0.15)$

### Pit Stop Noise

$$
L_{pit}' = L_{pit} + \epsilon_{pit}
$$

Where $\epsilon_{pit} \sim \mathcal{N}(0, 0.5)$

---

## 8. Strategy Evaluation Metrics

### Expected Position

$$
\mathbb{E}[pos] = \sum_{k=1}^{20} k \times P(pos = k)
$$

### Risk Score

$$
R = \alpha \times n_{stops} + \beta \times \text{Var}(pos)
$$

Where:
- $\alpha = 0.1$ (pit stop risk weight)
- $\beta = 1.0$ (position variance weight)

### Undercut Gain Estimate

$$
\Delta T_{undercut} \approx (age_{tire}^{rival} - age_{tire}^{new}) \times r_{deg} - L_{pit}
$$

**Example:**
- Rival tire age: 25 laps (Medium)
- New tire age: 0 laps
- Pit loss: 22 s

$$
\Delta T_{undercut} = (25 - 0) \times 0.03 - 22 = 0.75 - 22 = -21.25 \text{ s (net loss)}
$$

Undercut is beneficial if rival's tire advantage exceeds pit loss within overtaking window.

---

## 9. Confidence Intervals

### 95% Confidence Interval for Win Probability

$$
\text{CI}_{95}(P_{win}) = P_{win} \pm 1.96 \times \sqrt{\frac{P_{win}(1 - P_{win})}{N}}
$$

**Example:**
- $P_{win} = 0.35$
- $N = 1000$ runs

$$
\text{CI}_{95} = 0.35 \pm 1.96 \times \sqrt{\frac{0.35 \times 0.65}{1000}} = 0.35 \pm 0.030
$$

Result: $[0.320, 0.380]$

---

## 10. Required Monte Carlo Runs

### Sample Size Formula

$$
N = \left(\frac{z \times \sigma}{E}\right)^2
$$

Where:
- $z$ = Z-score for confidence level (1.96 for 95%)
- $\sigma$ = Standard deviation (estimated from pilot runs)
- $E$ = Margin of error (e.g., 0.02)

**Example:**
- Target: 95% confidence, ±2% margin
- Estimated $\sigma = 0.15$

$$
N = \left(\frac{1.96 \times 0.15}{0.02}\right)^2 = (14.7)^2 \approx 216 \text{ runs}
$$

---

## 11. Parallel Speedup

### Amdahl's Law

$$
S(p) = \frac{1}{(1-\alpha) + \frac{\alpha}{p}}
$$

Where:
- $p$ = Number of workers
- $\alpha$ = Parallelizable fraction (≈0.95 for MC)

**Example:**
- 8 workers
- $\alpha = 0.95$

$$
S(8) = \frac{1}{0.05 + \frac{0.95}{8}} = \frac{1}{0.169} \approx 5.9\times
$$

---

## 12. Strategy Tree Pruning

### Alpha-Beta Pruning Condition

Prune branch if:

$$
T_{branch} > T_{best} + \theta
$$

Where:
- $T_{branch}$ = Estimated time for branch
- $T_{best}$ = Best time found so far
- $\theta$ = Pruning threshold (5 seconds)

---

## Notation Summary

| Symbol | Meaning |
|--------|---------|
| $T_{lap}$ | Lap time (seconds) |
| $age_{tire}$ | Tire age (laps) |
| $r_{deg}$ | Degradation rate (s/lap) |
| $m_{fuel}$ | Fuel load (kg) |
| $P_{SC}$ | Safety car probability |
| $L_{pit}$ | Pit stop time loss (s) |
| $pos$ | Position (1-20) |
| $N$ | Number of Monte Carlo runs |
| $\epsilon$ | Random noise term |
| $\mathbb{E}[\cdot]$ | Expected value |
| $\text{Var}(\cdot)$ | Variance |

---

## References

1. "Optimal Race Strategy for Formula 1" - PhD Thesis, Cambridge, 2019
2. "Tire Degradation Modeling" - SAE Technical Paper 2020-01-0643
3. "Monte Carlo Methods in Motorsport" - Springer, 2021
4. Internal F1 telemetry data analysis (2015-2024)
