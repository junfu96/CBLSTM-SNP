# CBLSTM-SNP
Theoretical Derivation
# Theoretical Interpretation of CBLSTM-SNP

## Motivation

The original LSTM-SNP model performs recurrent state evolution through memory preservation and spike-driven consumption. Although this mechanism can model nonlinear temporal dynamics, excessive state attenuation may occur during long-term propagation, resulting in information loss and unstable hidden-state evolution.

To alleviate this issue, CBLSTM-SNP introduces an explicit compensation mechanism that adaptively restores useful state information during recurrent updates.

The hidden-state update is defined as

$$
u(t)=r(t)\odot u(t-1)-c(t)\odot a(t)+k(t)\odot z(t),
$$

where:

* $r(t)$: memory retention gate
* $c(t)$: spike consumption gate
* $k(t)$: compensation gate
* $z(t)$: regulation signal

The entire recurrent process can therefore be interpreted as a balance among memory preservation, information consumption, and adaptive restoration.

---

# Dynamic Interpretation

The hidden-state evolution of CBLSTM-SNP consists of three interacting components.

## 1. Memory Retention

$$
r(t)\odot u(t-1)
$$

This term preserves useful historical information.

A larger value of $r(t)$ allows more previous-state information to be carried into the next time step, improving long-term memory capability.

Conceptually:

```text
Previous State
      |
      v
 Retention Gate
      |
      v
Preserved Memory
```

---

## 2. Nonlinear Consumption

$$
c(t)\odot a(t)
$$

This term removes obsolete or redundant information.

The spike generation signal $a(t)$ acts as a nonlinear consumption factor that suppresses unnecessary state components during temporal propagation.

Conceptually:

```text
Current Memory
      |
      v
Consumption Gate
      |
      v
Information Reduction
```

---

## 3. Adaptive Compensation

$$
k(t)\odot z(t)
$$

This is the core innovation of CBLSTM-SNP.

When excessive attenuation occurs, the compensation pathway restores useful information back into the hidden state.

Unlike conventional recurrent architectures that only perform memory preservation and forgetting, CBLSTM-SNP explicitly introduces a recovery mechanism.

Conceptually:

```text
State Decay
     |
     v
Compensation Gate
     |
     v
State Restoration
```

---

# Overall Dynamical View

The complete state transition can be interpreted as:

```text
Historical State
        |
        v
   Memory Retention
        |
        v
-----------------------
|                     |
|  Spike Consumption |
|                     |
-----------------------
        |
        v
 Adaptive Compensation
        |
        v
   New Hidden State
```

Therefore, CBLSTM-SNP can be viewed as a compensation-balanced dynamical system.

---

# Lemma 1: Boundedness of Hidden States

## Research Question

A natural concern is:

> Will the compensation mechanism continuously increase the hidden state and eventually cause numerical explosion?

The boundedness lemma provides a theoretical answer.

---

## Main Result

The hidden state satisfies

$$
|u(t)|
\le
\gamma^t |u(0)|
+
\frac{M_a+\sqrt d}{1-\gamma},
$$

where

$$
0<\gamma<1.
$$

---

## Interpretation

The first term

$$
\gamma^t|u(0)|
$$

represents the influence of the initial state.

Since

$$
0<\gamma<1,
$$

the quantity

$$
\gamma^t
$$

gradually approaches zero as time increases.

Consequently, the effect of the initial condition diminishes over time.

---

The second term

$$
\frac{M_a+\sqrt d}{1-\gamma}
$$

is a finite constant.

Therefore, the hidden state can never grow without bound.

---

## Practical Meaning

This lemma guarantees that:

* hidden states remain finite;
* compensation does not cause state explosion;
* long-term recurrent computation remains numerically stable.

---

# Theorem 2: Multi-step Lipschitz Stability

## Research Question

Another important question is:

> If the input changes slightly, will the hidden state change dramatically?

A reliable forecasting model should produce only limited output variation under small input perturbations.

---

## State Difference

Consider two trajectories:

$$
u(t)
$$

and

$$
\tilde u(t).
$$

Define

$$
\Delta u(t)=u(t)-\tilde u(t),
$$

and

$$
\Delta x(t)=x(t)-\tilde x(t).
$$

Here:

* $\Delta u(t)$ measures the difference between hidden states.
* $\Delta x(t)$ measures the difference between inputs.

---

## Main Result

The theorem proves

$$
|\Delta u(t)|
\le
\alpha |\Delta u(t-1)|
+
\beta |\Delta x(t)|.
$$

---

# Physical Meaning of α

The parameter

$$
\alpha
$$

measures how strongly previous-state errors propagate through time.

### Case 1: α < 1

Errors gradually decrease.

```text
100
 ↓
80
 ↓
64
 ↓
51
 ↓
41
```

The system is stable.

---

### Case 2: α = 1

Errors remain unchanged.

```text
100
 ↓
100
 ↓
100
```

The system is neutrally stable.

---

### Case 3: α > 1

Errors grow continuously.

```text
100
 ↓
120
 ↓
144
 ↓
173
```

The system becomes unstable.

---

Therefore,

$$
\boxed{\alpha<1}
$$

is the key condition for long-term stability.

---

# Physical Meaning of β

The parameter

$$
\beta
$$

measures the sensitivity of the model to input perturbations.

A smaller value of $\beta$ indicates:

* stronger robustness;
* better resistance to noise;
* smoother prediction behavior.

---

# Multi-step Stability

The theorem further establishes

$$
|\Delta u(T)|
\le
\alpha^T|\Delta u(0)|
+
\frac{1-\alpha^T}{1-\alpha}
\beta
\sup_{t\le T}|\Delta x(t)|.
$$

This result implies:

1. Initial-state errors gradually vanish.
2. Input perturbations remain bounded.
3. Hidden-state trajectories remain stable over long forecasting horizons.

---

# Why Compensation Does Not Destroy Stability

The compensation mechanism contributes additional terms to both

$$
\alpha
$$

and

$$
\beta.
$$

However:

* all compensation gates are bounded;
* sigmoid-based gates have small derivatives;
* compensation signals are constrained.

Therefore, the compensation pathway introduces only limited perturbation amplification.

Instead, it helps prevent excessive information loss and improves the smoothness of hidden-state evolution.

---

# Overall Theoretical Conclusion

The theoretical analysis establishes three important properties of CBLSTM-SNP.

## Dynamic Balance

Hidden-state evolution is governed by the interaction of:

* memory retention,
* spike consumption,
* adaptive compensation.

## State Boundedness

Hidden states remain uniformly bounded and cannot diverge during long-term recurrent computation.

## Lipschitz Stability

Small perturbations in inputs lead to bounded changes in hidden states, ensuring robust and stable temporal modeling.

Together, these results provide theoretical support for the effectiveness and reliability of the proposed compensation-balanced recurrent architecture.
