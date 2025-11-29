# Feature Engineering Formulas

Mathematical documentation for all feature calculations with LaTeX notation, examples, edge cases, and validation rules.

## Table of Contents

1. [Pace & Timing](#pace--timing)
2. [Degradation Models](#degradation-models)
3. [Pitstop Strategy](#pitstop-strategy)
4. [Tire Performance](#tire-performance)
5. [Weather Corrections](#weather-corrections)
6. [Safety Car Probability](#safety-car-probability)
7. [Traffic Impact](#traffic-impact)
8. [Telemetry Metrics](#telemetry-metrics)
9. [Utility Functions](#utility-functions)

---

## Pace & Timing

### Lap Pace Delta

**Formula:**

$$
\Delta_{\text{leader}} = t_{\text{driver}} - t_{\text{leader}}
$$

$$
\Delta_{\text{teammate}} = t_{\text{driver}} - t_{\text{teammate}}
$$

$$
\Delta_{\text{avg}} = t_{\text{driver}} - \frac{1}{n}\sum_{i=1}^{n} t_i
$$

**Variables:**
- $t_{\text{driver}}$: Driver's lap time (seconds)
- $t_{\text{leader}}$: Race leader's lap time (seconds)
- $t_{\text{teammate}}$: Teammate's lap time (seconds)
- $n$: Number of drivers in session

**Examples:**

| Driver Time | Leader Time | Teammate Time | Avg Time | $\Delta_{\text{leader}}$ | $\Delta_{\text{teammate}}$ | $\Delta_{\text{avg}}$ |
|-------------|-------------|---------------|----------|--------------------------|----------------------------|-----------------------|
| 78.5s | 77.2s | 78.8s | 79.1s | +1.3s | -0.3s | -0.6s |
| 82.1s | 77.2s | 78.8s | 79.1s | +4.9s | +3.3s | +3.0s |

**Edge Cases:**
- **Missing leader data:** Use session best as reference
- **No teammate:** Set $\Delta_{\text{teammate}}$ to NaN
- **First lap:** Exclude from average calculation

**Validation:**
- $|\Delta| < 30$ seconds (sanity check)
- Non-null lap times for driver and reference

---

### Rolling Pace

**Formula:**

$$
\text{RollingPace}_t = \frac{1}{w} \sum_{i=t-w+1}^{t} t_i
$$

$$
\text{Volatility}_t = \sqrt{\frac{1}{w-1} \sum_{i=t-w+1}^{t} (t_i - \text{RollingPace}_t)^2}
$$

**Variables:**
- $w$: Window size (typically 3, 5, or 10 laps)
- $t_i$: Lap time at lap $i$
- $t$: Current lap number

**Examples:**

| Lap | Time | 3-Lap Rolling | 5-Lap Rolling | Volatility (3-lap) |
|-----|------|---------------|---------------|---------------------|
| 1 | 78.5 | N/A | N/A | N/A |
| 2 | 78.3 | N/A | N/A | N/A |
| 3 | 78.7 | 78.50 | N/A | 0.20 |
| 4 | 78.9 | 78.63 | N/A | 0.31 |
| 5 | 79.1 | 78.90 | 78.70 | 0.20 |

**Edge Cases:**
- **Insufficient laps:** Return NaN until window is full
- **Outlier laps:** Remove outliers (>3σ) before calculation
- **Pit laps:** Exclude from rolling window

**Validation:**
- $w \geq 3$ (minimum window size)
- At least $w$ laps available

---

### Sector Pace

**Formula:**

$$
\text{SectorAvg}_s = \frac{1}{n} \sum_{i=1}^{n} s_i
$$

$$
\text{Consistency}_s = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (s_i - \text{SectorAvg}_s)^2}
$$

$$
\text{Percentile}_s = \frac{\text{rank}(s_{\text{driver}})}{N} \times 100
$$

**Variables:**
- $s_i$: Sector time for lap $i$
- $n$: Number of laps
- $N$: Total number of drivers

**Examples:**

| Sector | Avg | Best | Consistency | Percentile |
|--------|-----|------|-------------|------------|
| 1 | 25.3s | 24.9s | 0.4s | 65% |
| 2 | 32.1s | 31.5s | 0.6s | 72% |
| 3 | 21.1s | 20.8s | 0.3s | 58% |

**Edge Cases:**
- **Missing sector:** Use lap time / 3 as estimate
- **First lap:** Exclude from statistics
- **Yellow flags:** Mark sector as invalid

**Validation:**
- $\text{SectorAvg}_1 + \text{SectorAvg}_2 + \text{SectorAvg}_3 \approx \text{LapTime}$ (within 1s)
- Sector times > 0

---

## Degradation Models

### Linear Degradation

**Formula:**

$$
t(\text{age}) = \beta_0 + \beta_1 \cdot \text{age}
$$

$$
\beta_1 = \frac{\sum_{i=1}^{n}(\text{age}_i - \bar{\text{age}})(t_i - \bar{t})}{\sum_{i=1}^{n}(\text{age}_i - \bar{\text{age}})^2}
$$

$$
R^2 = 1 - \frac{\sum_{i=1}^{n}(t_i - \hat{t}_i)^2}{\sum_{i=1}^{n}(t_i - \bar{t})^2}
$$

**Variables:**
- $t$: Lap time (seconds)
- $\text{age}$: Tire age (laps)
- $\beta_0$: Intercept (base lap time)
- $\beta_1$: Degradation rate (seconds per lap)
- $R^2$: Coefficient of determination

**Examples:**

| Tire Age | Lap Time | Predicted | Residual |
|----------|----------|-----------|----------|
| 1 | 78.5 | 78.4 | +0.1 |
| 5 | 79.2 | 79.2 | 0.0 |
| 10 | 80.5 | 80.4 | +0.1 |
| 15 | 81.8 | 81.6 | +0.2 |
| 20 | 83.2 | 82.8 | +0.4 |

**Result:** $\beta_1 = 0.24$ s/lap, $R^2 = 0.98$

**Edge Cases:**
- **Insufficient laps:** Require $n \geq 5$ for regression
- **Negative slope:** Set degradation to 0 (improving pace)
- **Poor fit:** If $R^2 < 0.3$, mark as invalid

**Validation:**
- $0 \leq \beta_1 \leq 2.0$ (reasonable degradation range)
- $R^2 > 0.3$ (minimum fit quality)
- $n \geq 5$ laps

---

### Exponential Degradation

**Formula:**

$$
\text{deg}(t) = A \cdot e^{B \cdot t}
$$

**Fitted by minimizing:**

$$
\min_{A, B} \sum_{i=1}^{n} \left( \Delta t_i - A \cdot e^{B \cdot \text{age}_i} \right)^2
$$

**Variables:**
- $A$: Exponential coefficient (initial degradation)
- $B$: Exponential growth rate
- $t$: Tire age (laps)
- $\Delta t_i$: Lap time delta vs first lap

**Examples:**

| Tire Age | Lap Delta | Predicted (A=0.1, B=0.15) |
|----------|-----------|----------------------------|
| 1 | 0.0 | 0.12 |
| 5 | 0.8 | 0.21 |
| 10 | 2.1 | 0.45 |
| 15 | 4.8 | 0.96 |
| 20 | 10.2 | 2.01 |

**Result:** $A = 0.1$, $B = 0.15$

**Edge Cases:**
- **Negative B:** Indicates improving pace, set to linear model
- **Very high B:** Cap at 0.5 to avoid numerical instability
- **Convergence failure:** Fall back to linear model

**Validation:**
- $0 < A < 5$ (reasonable initial degradation)
- $0 < B < 0.5$ (reasonable growth rate)
- Converged within 1000 iterations

---

### Cliff Detection

**Formula:**

$$
\text{CliffDetected} = \begin{cases}
1 & \text{if } \exists k: \Delta t_k - \Delta t_{k-1} > \theta \text{ for } c \text{ consecutive laps} \\
0 & \text{otherwise}
\end{cases}
$$

$$
\text{CliffMagnitude} = \max_{k} (\Delta t_k - \Delta t_{k-1})
$$

**Variables:**
- $\theta$: Threshold for cliff detection (typically 1.0s)
- $c$: Consecutive laps above threshold (typically 3)
- $\Delta t_k$: Lap time delta vs reference

**Examples:**

**No Cliff:**
| Lap | Time | Delta | Cliff? |
|-----|------|-------|--------|
| 10 | 78.5 | +0.0 | No |
| 11 | 78.8 | +0.3 | No |
| 12 | 79.2 | +0.4 | No |
| 13 | 79.6 | +0.4 | No |

**Cliff Detected:**
| Lap | Time | Delta | Cliff? |
|-----|------|-------|--------|
| 10 | 78.5 | +0.0 | No |
| 11 | 79.8 | +1.3 | Maybe |
| 12 | 81.2 | +1.4 | Maybe |
| 13 | 82.5 | +1.3 | **Yes** |

**Edge Cases:**
- **Pit lap:** Exclude from cliff detection
- **Yellow flags:** Exclude from cliff detection
- **First stint:** Require at least 5 laps before detecting

**Validation:**
- $\theta > 0.5$ (meaningful threshold)
- $c \geq 2$ (consecutive laps required)

---

### Degradation Anomaly

**Formula:**

$$
z_i = \frac{\Delta t_i - \mu}{\sigma}
$$

$$
\text{Anomaly}_i = \begin{cases}
1 & \text{if } |z_i| > \theta_z \\
0 & \text{otherwise}
\end{cases}
$$

**Variables:**
- $z_i$: Z-score for lap $i$
- $\mu$: Mean degradation
- $\sigma$: Standard deviation of degradation
- $\theta_z$: Z-score threshold (typically 3.0)

**Examples:**

| Lap | Delta | Mean | Std | Z-Score | Anomaly? |
|-----|-------|------|-----|---------|----------|
| 5 | 0.5 | 0.6 | 0.2 | -0.5 | No |
| 10 | 1.2 | 1.1 | 0.2 | +0.5 | No |
| 15 | 3.8 | 1.6 | 0.3 | **+7.3** | **Yes** |
| 20 | 2.1 | 2.1 | 0.3 | 0.0 | No |

**Edge Cases:**
- **Insufficient data:** Require $n \geq 10$ laps
- **High variability:** Increase $\theta_z$ if $\sigma > 0.5$
- **Systematic drift:** Use rolling window for $\mu$ and $\sigma$

**Validation:**
- $\theta_z \geq 2.0$ (reasonable threshold)
- $n \geq 10$ laps for statistics

---

## Pitstop Strategy

### Undercut Delta

**Formula:**

$$
\Delta_{\text{undercut}} = t_{\text{gained}} - t_{\text{pit}} - \sum_{i=1}^{w} p_{\text{warmup},i}
$$

$$
t_{\text{gained}} = (t_{\text{opponent},\text{old}} - t_{\text{self},\text{new}}) \times n_{\text{laps}}
$$

$$
p_{\text{warmup},i} = p_0 \cdot e^{-i/\tau}
$$

**Variables:**
- $t_{\text{pit}}$: Pit loss time (track-specific, typically 20-25s)
- $p_{\text{warmup},i}$: Warmup penalty for lap $i$ (typically 0.3s)
- $w$: Warmup laps (typically 2)
- $\tau$: Warmup decay constant

**Examples:**

**Successful Undercut:**
- Opponent old tire pace: 80.0s
- Self new tire pace: 78.5s
- Pace delta: 1.5s per lap
- Pit loss: 23s
- Warmup penalty: 2 × 0.3s = 0.6s
- Laps to pit opponent: 3
- **Net undercut:** $1.5 \times 3 - 23 - 0.6 = -19.1s$ → Undercut fails by 19s

**Marginal Undercut:**
- Pace delta: 8.0s per lap
- Laps to pit: 3
- **Net undercut:** $8.0 \times 3 - 23 - 0.6 = +0.4s$ → Undercut succeeds by 0.4s

**Edge Cases:**
- **Traffic:** Reduce $t_{\text{gained}}$ by traffic penalty
- **VSC pit:** Reduce $t_{\text{pit}}$ by ~50%
- **Long warmup:** Increase $w$ for cold conditions

**Validation:**
- $15 < t_{\text{pit}} < 35$ (reasonable pit loss range)
- $0 < p_{\text{warmup}} < 1.0$ (reasonable warmup penalty)
- $1 \leq w \leq 5$ (reasonable warmup laps)

---

### Overcut Delta

**Formula:**

$$
\Delta_{\text{overcut}} = t_{\text{gained}} - t_{\text{offset}}
$$

$$
t_{\text{gained}} = f_{\text{fuel}} \times n_{\text{laps}} + t_{\text{tire}} \times n_{\text{laps}}
$$

$$
f_{\text{fuel}} = \alpha \cdot m_{\text{fuel}} \quad (\text{typically } \alpha = 0.03 \text{ s/kg})
$$

**Variables:**
- $f_{\text{fuel}}$: Fuel effect (lighter car advantage)
- $t_{\text{tire}}$: Tire offset advantage (fresher tires after opponent pits)
- $t_{\text{offset}}$: Risk of losing track position

**Examples:**

**Successful Overcut:**
- Fuel advantage: 0.03s/kg × 10kg = 0.3s per lap
- Tire offset: 0.2s per lap
- Total advantage: 0.5s per lap
- Laps staying out: 5
- **Net overcut:** $0.5 \times 5 = +2.5s$ → Overcut succeeds

**Failed Overcut:**
- Fuel + tire advantage: 0.3s per lap
- Laps staying out: 3
- Traffic penalty: 0.5s per lap
- **Net overcut:** $(0.3 - 0.5) \times 3 = -0.6s$ → Overcut fails

**Edge Cases:**
- **Heavy traffic:** Overcut advantage negated
- **Safety car:** Reset overcut calculation
- **Tire cliff:** Accelerate overcut failure

**Validation:**
- $0.01 < \alpha < 0.05$ (reasonable fuel effect)
- $0 < t_{\text{tire}} < 1.0$ (reasonable tire offset)

---

### Pit Loss Model

**Formula:**

$$
t_{\text{pit,total}} = t_{\text{pit,static}} + t_{\text{entry}} + t_{\text{exit}}
$$

$$
t_{\text{entry}} = \frac{L_{\text{entry}}}{v_{\text{entry}}} - \frac{L_{\text{entry}}}{v_{\text{race}}}
$$

$$
t_{\text{exit}} = \frac{L_{\text{exit}}}{v_{\text{exit}}} - \frac{L_{\text{exit}}}{v_{\text{race}}}
$$

**Variables:**
- $t_{\text{pit,static}}$: Stationary time in pit box (typically 2-3s)
- $L_{\text{entry}}$: Pit entry length (meters)
- $L_{\text{exit}}$: Pit exit length (meters)
- $v_{\text{entry}}$, $v_{\text{exit}}$: Pit lane speed limit (typically 80 km/h)
- $v_{\text{race}}$: Average race speed

**Examples:**

**Monaco (short pit lane):**
- Static: 2.5s
- Entry loss: 3.2s
- Exit loss: 3.8s
- **Total:** 9.5s

**Silverstone (long pit lane):**
- Static: 2.5s
- Entry loss: 8.5s
- Exit loss: 9.2s
- **Total:** 20.2s

**Monza (very long pit lane):**
- Static: 2.5s
- Entry loss: 12.3s
- Exit loss: 13.7s
- **Total:** 28.5s

**Edge Cases:**
- **VSC:** Reduce loss to ~50% of normal
- **Safety car:** Pit loss near zero (if timed perfectly)
- **Unsafe release:** Add penalty time

**Validation:**
- $5 < t_{\text{pit,total}} < 40$ (reasonable range across all tracks)
- $1 < t_{\text{pit,static}} < 5$ (reasonable stationary time)

---

### Pit Window

**Formula:**

$$
W_{\text{optimal}} = \left[ r_{\text{min}} \cdot L_{\text{race}}, r_{\text{max}} \cdot L_{\text{race}} \right]
$$

$$
p_{\text{early}} = \max(0, r_{\text{min}} \cdot L_{\text{race}} - l_{\text{current}}) \times \alpha
$$

$$
p_{\text{late}} = \max(0, l_{\text{current}} - r_{\text{max}} \cdot L_{\text{race}}) \times \beta
$$

**Variables:**
- $r_{\text{min}}, r_{\text{max}}$: Optimal window bounds (typically 0.4, 0.6)
- $L_{\text{race}}$: Total race laps
- $l_{\text{current}}$: Current lap
- $\alpha, \beta$: Early/late penalty factors

**Examples:**

**Monaco (78 laps):**
- Optimal window: [31, 47]
- Early stop (lap 25): Penalty = $(31 - 25) \times 1.0 = 6.0s$
- Late stop (lap 55): Penalty = $(55 - 47) \times 2.0 = 16.0s$

**Edge Cases:**
- **Two-stop strategy:** Split window into two segments
- **Safety car:** Adjust window dynamically
- **Tire degradation:** Shift window earlier if high deg

**Validation:**
- $0 < r_{\text{min}} < r_{\text{max}} < 1$ (valid range)
- $\alpha, \beta > 0$ (positive penalties)

---

## Tire Performance

### Tire Warmup

**Formula:**

$$
\Delta t_{\text{warmup}} = \max(0, t_{\text{target}} - T_{\text{tire}}) \times \gamma
$$

$$
T_{\text{tire}}(l) = T_{\text{ambient}} + (T_{\text{optimal}} - T_{\text{ambient}}) \cdot (1 - e^{-l/\tau})
$$

**Variables:**
- $T_{\text{tire}}$: Tire temperature (°C)
- $T_{\text{optimal}}$: Optimal tire temperature (typically 95-105°C)
- $\tau$: Warmup time constant (laps)
- $\gamma$: Temperature-pace coefficient

**Examples:**

| Lap | Tire Temp | Delta from Optimal | Time Penalty |
|-----|-----------|---------------------|--------------|
| 1 | 60°C | -40°C | +0.8s |
| 2 | 85°C | -15°C | +0.3s |
| 3 | 95°C | -5°C | +0.1s |
| 4 | 100°C | 0°C | 0.0s |

**Edge Cases:**
- **Hot track:** Reduce warmup laps
- **Cold conditions:** Increase warmup laps
- **Safety car restart:** Reset warmup model

**Validation:**
- $50 < T_{\text{optimal}} < 120$ (reasonable optimal temp)
- $0 < \tau < 5$ (reasonable warmup laps)

---

### Tire Dropoff

**Formula:**

$$
\text{Dropoff} = \begin{cases}
1 & \text{if } \frac{t_l - t_{\text{ref}}}{t_{\text{ref}}} > \theta_{\text{cliff}} \\
0 & \text{otherwise}
\end{cases}
$$

$$
l_{\text{dropoff}} = \arg\max_{l} \left( t_l - t_{l-1} \right)
$$

**Variables:**
- $\theta_{\text{cliff}}$: Cliff threshold (typically 5% degradation)
- $t_l$: Lap time at lap $l$
- $t_{\text{ref}}$: Reference lap time (typically best in stint)

**Examples:**

**Soft Tire Dropoff (Monaco):**
| Lap | Time | Delta | Dropoff? |
|-----|------|-------|----------|
| 10 | 78.5 | +0.0% | No |
| 15 | 79.2 | +0.9% | No |
| 20 | 80.5 | +2.5% | No |
| 25 | 82.8 | **+5.5%** | **Yes** |

**Hard Tire (no dropoff):**
| Lap | Time | Delta | Dropoff? |
|-----|------|-------|----------|
| 10 | 79.0 | +0.0% | No |
| 20 | 79.8 | +1.0% | No |
| 30 | 80.5 | +1.9% | No |
| 40 | 81.2 | +2.8% | No |

**Edge Cases:**
- **Traffic:** Exclude traffic-affected laps
- **Yellow flags:** Exclude from dropoff detection
- **Pit window:** Consider dropoff in pit timing

**Validation:**
- $0.03 < \theta_{\text{cliff}} < 0.10$ (reasonable cliff threshold)
- At least 10 laps to detect dropoff

---

## Weather Corrections

### Weather-Adjusted Pace

**Formula:**

$$
t_{\text{adjusted}} = t_{\text{actual}} - \left( \alpha_T \Delta T + \alpha_R \cdot R + \alpha_W \cdot W \right)
$$

$$
\Delta T = \beta \cdot (T_{\text{track}} - T_{\text{ref}}) + (1-\beta) \cdot (T_{\text{air}} - T_{\text{ref,air}})
$$

**Variables:**
- $\alpha_T$: Temperature coefficient (typically 0.002 s/°C)
- $\alpha_R$: Rain coefficient (typically 0.05 s/mm)
- $\alpha_W$: Wind coefficient (typically 0.01 s/(km/h))
- $\beta$: Track temperature weight (typically 0.7)

**Examples:**

| Condition | Track Temp | Rain | Wind | Correction |
|-----------|------------|------|------|------------|
| Hot & Dry | +10°C | 0mm | 5 km/h | +0.07s |
| Cool | -5°C | 0mm | 10 km/h | -0.09s |
| Light Rain | +0°C | 2mm | 15 km/h | +0.25s |
| Heavy Rain | +0°C | 10mm | 20 km/h | +0.70s |

**Edge Cases:**
- **Extreme heat:** Cap correction at +0.5s
- **Extreme cold:** Cap correction at -0.5s
- **Mixed conditions:** Use lap-by-lap corrections

**Validation:**
- $0.001 < \alpha_T < 0.005$ (reasonable temp effect)
- $0.02 < \alpha_R < 0.10$ (reasonable rain effect)
- $0.005 < \alpha_W < 0.02$ (reasonable wind effect)

---

### Track Evolution

**Formula:**

$$
f_{\text{evolution}} = 1 - r \cdot t_{\text{session}} \cdot (1 - e^{-t_{\text{session}}/\tau})
$$

$$
t_{\text{adjusted}} = \frac{t_{\text{actual}}}{f_{\text{evolution}}}
$$

**Variables:**
- $r$: Rubber buildup rate (typically 0.05% per hour)
- $t_{\text{session}}$: Session elapsed time (hours)
- $\tau$: Evolution saturation constant
- $f_{\text{evolution}}$: Track evolution factor (≤1)

**Examples:**

| Session Time | Evolution Factor | Lap Time Improvement |
|--------------|------------------|----------------------|
| 0 hours | 1.000 | 0.0% |
| 1 hour | 0.995 | 0.5% (~0.4s @ 80s lap) |
| 2 hours | 0.990 | 1.0% (~0.8s) |
| 3 hours | 0.985 | 1.5% (~1.2s) |

**Edge Cases:**
- **Rain:** Reset evolution factor to 1.0
- **Long session:** Cap improvement at 2%
- **New track surface:** Increase $r$ to 0.10

**Validation:**
- $0.02 < r < 0.10$ (reasonable evolution rate)
- $0.98 < f_{\text{evolution}} < 1.0$ (reasonable improvement)

---

## Safety Car Probability

### Historical Probability

**Formula:**

$$
P_{\text{SC}} = \frac{N_{\text{SC}}}{N_{\text{races}}}
$$

$$
P_{\text{track}} = w_1 \cdot P_{\text{SC,track}} + w_2 \cdot P_{\text{SC,global}}
$$

**Variables:**
- $N_{\text{SC}}$: Number of races with safety car
- $N_{\text{races}}$: Total number of races
- $w_1, w_2$: Track-specific and global weights

**Examples:**

| Track | SC Races | Total Races | SC Probability |
|-------|----------|-------------|----------------|
| Monaco | 18 | 20 | **90%** |
| Spa | 12 | 20 | 60% |
| Bahrain | 8 | 20 | 40% |
| Austria | 4 | 20 | **20%** |

**Edge Cases:**
- **New track:** Use global average + weather adjustment
- **Track changes:** Weight recent races higher
- **Sprint weekend:** Adjust for shorter race

**Validation:**
- $0 < P_{\text{SC}} < 1$ (valid probability)
- $N_{\text{races}} \geq 5$ (sufficient sample size)

---

### Real-Time Probability

**Formula:**

$$
P_{\text{RT}}(t) = P_{\text{base}} + \sum_{i} w_i \cdot I_i(t) \cdot e^{-(t - t_i)/\tau}
$$

$$
I_i(t) = \begin{cases}
0.8 & \text{crash} \\
0.4 & \text{mechanical} \\
0.3 & \text{debris} \\
0.5 & \text{weather}
\end{cases}
$$

**Variables:**
- $P_{\text{base}}$: Base historical probability
- $w_i$: Incident weights
- $I_i(t)$: Incident indicator at time $t$
- $\tau$: Probability decay constant (laps)

**Examples:**

| Lap | Incident | Weight | Decay | Real-Time P |
|-----|----------|--------|-------|-------------|
| 10 | Crash | 0.8 | 1.0 | **0.65** |
| 11 | - | - | 0.8 | 0.52 |
| 12 | - | - | 0.6 | 0.42 |
| 13 | Debris | 0.3 | 0.5 | 0.45 |

**Edge Cases:**
- **Multiple incidents:** Sum weighted probabilities (cap at 1.0)
- **Long gap:** Decay to base probability after 10 laps
- **VSC first:** Increase SC probability by 20%

**Validation:**
- $0 < P_{\text{RT}} < 1$ (valid probability)
- $\tau > 0$ (positive decay)

---

## Traffic Impact

### Clean Air Penalty

**Formula:**

$$
p_{\text{traffic}} = p_0 \cdot e^{-g/d} - b_{\text{DRS}}
$$

$$
p_0 = 0.4 \text{ s}, \quad d = 1.0 \text{ s}, \quad b_{\text{DRS}} = 0.3 \text{ s}
$$

**Variables:**
- $p_0$: Base traffic penalty (seconds)
- $g$: Gap to car ahead (seconds)
- $d$: Decay distance (seconds)
- $b_{\text{DRS}}$: DRS benefit (seconds)

**Examples:**

| Gap | DRS? | Penalty |
|-----|------|---------|
| 0.5s | No | +0.24s |
| 1.0s | No | +0.15s |
| 2.0s | No | +0.05s |
| 0.8s | Yes | -0.11s (DRS benefit > penalty) |
| 5.0s | No | +0.00s |

**Edge Cases:**
- **Lapping:** Use reduced penalty (50%)
- **DRS train:** Penalty increases in middle of train
- **Wet conditions:** Increase penalty by 50%

**Validation:**
- $0.2 < p_0 < 0.8$ (reasonable base penalty)
- $0.5 < d < 2.0$ (reasonable decay distance)
- $0.2 < b_{\text{DRS}} < 0.5$ (reasonable DRS benefit)

---

### Traffic Density

**Formula:**

$$
\rho_{\text{traffic}}(t) = \sum_{i \in W} \mathbb{1}_{|g_i| < \theta}
$$

$$
W = \{i : |p_i - p_{\text{driver}}| \leq k\}
$$

**Variables:**
- $\rho_{\text{traffic}}$: Traffic density (number of cars)
- $W$: Window of $k$ positions (typically 5)
- $g_i$: Gap to car $i$ (seconds)
- $\theta$: Traffic threshold (typically 3.0s)

**Examples:**

| Position | Gap Ahead | Gap Behind | Window Cars | Density |
|----------|-----------|------------|-------------|---------|
| 5 | 1.2s | 0.8s | [3,4,5,6,7] | **4** |
| 10 | 5.2s | 4.8s | [8,9,10,11,12] | **0** |
| 15 | 0.5s | 2.1s | [13,14,15,16,17] | **3** |

**Edge Cases:**
- **Leaders:** Only count cars behind
- **Backmarkers:** Only count cars ahead
- **Safety car:** Density = all cars (field compressed)

**Validation:**
- $0 \leq \rho_{\text{traffic}} \leq k$ (valid density range)
- $\theta > 0$ (positive threshold)

---

## Telemetry Metrics

### Driver Aggression

**Formula:**

$$
A = w_T \cdot T_{\max} + w_B \cdot B_{\max} + w_S \cdot S_{\text{smooth}}
$$

$$
S_{\text{smooth}} = 1 - \frac{\sigma_{\text{steering}}}{\max(\sigma_{\text{steering}})}
$$

**Variables:**
- $T_{\max}$: Maximum throttle percentage
- $B_{\max}$: Maximum brake percentage
- $S_{\text{smooth}}$: Steering smoothness
- $w_T, w_B, w_S$: Weights (typically 0.4, 0.4, 0.2)

**Examples:**

| Driver | Throttle | Brake | Smoothness | Aggression |
|--------|----------|-------|------------|------------|
| A | 98% | 100% | 0.85 | **0.96** (aggressive) |
| B | 95% | 95% | 0.92 | 0.94 (balanced) |
| C | 92% | 90% | 0.95 | **0.91** (smooth) |

**Edge Cases:**
- **Wet conditions:** Lower aggression threshold
- **Qualifying:** Higher aggression expected
- **Tire management:** Lower aggression expected

**Validation:**
- $0 < A < 1$ (valid aggression score)
- $0 < T_{\max}, B_{\max} < 100$ (valid percentages)

---

### Fuel Effect

**Formula:**

$$
\Delta t_{\text{fuel}} = \alpha \cdot m_{\text{fuel}}
$$

$$
m_{\text{fuel}}(l) = m_0 \cdot \left(1 - \frac{l}{L}\right)
$$

**Variables:**
- $\alpha$: Fuel effect coefficient (typically 0.03 s/kg)
- $m_{\text{fuel}}$: Current fuel load (kg)
- $m_0$: Starting fuel load (typically 100-110 kg)
- $L$: Total race laps

**Examples:**

| Lap | Fuel Load | Fuel Effect |
|-----|-----------|-------------|
| 1 | 110 kg | +3.30s |
| 20 | 80 kg | +2.40s |
| 40 | 50 kg | +1.50s |
| 60 | 20 kg | +0.60s |

**Edge Cases:**
- **Fuel saving:** Actual consumption lower than model
- **Formation lap:** Add 2-3 kg to starting fuel
- **Safety car:** Reduce fuel consumption rate

**Validation:**
- $0.01 < \alpha < 0.05$ (reasonable fuel effect)
- $0 < m_{\text{fuel}} < 150$ (reasonable fuel load)

---

## Utility Functions

### Exponential Smoothing

**Formula:**

$$
S_t = \begin{cases}
y_1 & t = 1 \\
\alpha \cdot y_t + (1 - \alpha) \cdot S_{t-1} & t > 1
\end{cases}
$$

**Variables:**
- $S_t$: Smoothed value at time $t$
- $y_t$: Observed value at time $t$
- $\alpha$: Smoothing factor (0 < α < 1)

**Examples:**

| Time | Observed | Smoothed (α=0.3) | Smoothed (α=0.7) |
|------|----------|------------------|------------------|
| 1 | 78.5 | 78.50 | 78.50 |
| 2 | 79.2 | 78.71 | 78.99 |
| 3 | 78.9 | 78.77 | 78.93 |
| 4 | 80.1 | 79.17 | 79.76 |

**Edge Cases:**
- **High variability:** Use lower α (more smoothing)
- **Rapid changes:** Use higher α (less smoothing)
- **Missing values:** Skip and continue with last $S_t$

**Validation:**
- $0 < \alpha < 1$ (valid smoothing factor)

---

### Changepoint Detection (CUSUM)

**Formula:**

$$
S_t = \max(0, S_{t-1} + (y_t - \mu - k))
$$

$$
\text{Changepoint} = \begin{cases}
t & \text{if } S_t > h \\
\text{None} & \text{otherwise}
\end{cases}
$$

**Variables:**
- $S_t$: CUSUM statistic at time $t$
- $\mu$: Mean value
- $k$: Slack parameter (typically 0.5σ)
- $h$: Threshold (typically 5σ)

**Examples:**

**Changepoint Detected:**
| Time | Value | Mean | CUSUM | Changepoint? |
|------|-------|------|-------|--------------|
| 10 | 78.5 | 78.5 | 0.0 | No |
| 11 | 78.7 | 78.5 | 0.1 | No |
| 12 | 79.2 | 78.5 | 0.7 | No |
| 13 | 81.5 | 78.5 | **3.6** | **Yes** |

**Edge Cases:**
- **Multiple changepoints:** Reset CUSUM after detection
- **Gradual drift:** May not detect without cumulative effect
- **High noise:** Increase $k$ and $h$

**Validation:**
- $k > 0$ (positive slack)
- $h > 0$ (positive threshold)

---

**Last Updated:** 2024-12-20  
**Version:** 1.0.0  
**Authors:** F1 Race Strategy Intelligence Team
