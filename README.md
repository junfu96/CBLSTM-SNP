# CBLSTM-SNP
Theoretical Derivation
# Theoretical Interpretation of CBLSTM-SNP

## Motivation

The original LSTM-SNP model performs recurrent state evolution through memory preservation and spike-driven consumption. Although this mechanism is capable of modeling nonlinear temporal dynamics, excessive state attenuation may occur during long-term propagation, resulting in unstable hidden-state evolution and information loss.

To address this issue, CBLSTM-SNP introduces a compensation mechanism that explicitly restores useful state information during recurrent updates.

The hidden state update can be written as

[
u(t)
====

## r(t)\odot u(t-1)

c(t)\odot a(t)
+
k(t)\odot z(t),
]

where:

* (r(t)) controls memory retention;
* (c(t)) controls spike consumption;
* (k(t)) controls adaptive compensation;
* (z(t)) is the regulation signal.

The entire model can therefore be interpreted as a balance among memory preservation, information consumption, and adaptive restoration.

---

# Dynamic Interpretation

The recurrent dynamics consist of three interacting components.

## 1. Memory Retention

[
r(t)\odot u(t-1)
]

This term preserves useful historical information.

A larger value of (r(t)) allows more previous-state information to be carried into the next time step, thereby enhancing long-term memory capability.

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

[
c(t)\odot a(t)
]

This term removes obsolete or redundant information.

The spike-generation signal (a(t)) acts as a nonlinear consumption factor that suppresses unnecessary state components during temporal propagation.

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

[
k(t)\odot z(t)
]

This is the key innovation of CBLSTM-SNP.

When excessive attenuation occurs, the compensation pathway restores useful information back into the hidden state.

Unlike conventional recurrent models that only perform memory preservation and forgetting, CBLSTM-SNP explicitly introduces a recovery mechanism.

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

The complete state transition can be viewed as

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

Therefore, CBLSTM-SNP can be regarded as a compensation-balanced dynamical system.

---

# Lemma 1: Boundedness of Hidden States

## Question

A natural question is:

> Will the compensation mechanism continuously increase the hidden state and eventually cause numerical explosion?

The boundedness lemma answers this question.

---

## Main Result

The hidden state satisfies

[
|u(t)|
\le
\gamma^t |u(0)|
+
\frac{M_a+\sqrt d}{1-\gamma},
]

where

[
0<\gamma<1.
]

---

## Interpretation

The first term

[
\gamma^t|u(0)|
]

represents the influence of the initial state.

Since

[
0<\gamma<1,
]

we have

[
\gamma^t\rightarrow 0
]

as time increases.

Thus the effect of the initial condition gradually disappears.

---

The second term

[
\frac{M_a+\sqrt d}{1-\gamma}
]

is a finite constant.

This means that the hidden state can never grow without bound.

---

## Practical Meaning

The boundedness theorem guarantees that:

* hidden states remain finite;
* compensation does not cause explosion;
* long-term recurrent computation remains numerically stable.

---

# Theorem 2: Multi-step Lipschitz Stability

## Question

Another important issue is:

> If the input changes slightly, will the hidden state change dramatically?

A reliable forecasting model should produce only small output changes when the input perturbation is small.

---

## State Difference

Consider two trajectories:

[
u(t)
]

and

[
\tilde u(t).
]

Define

[
\Delta u(t)
===========

u(t)-\tilde u(t),
]

and

[
\Delta x(t)
===========

x(t)-\tilde x(t).
]

---

## Main Result

The theorem proves

[
|\Delta u(t)|
\le
\alpha
|\Delta u(t-1)|
+
\beta
|\Delta x(t)|.
]

---

# Physical Meaning of α

The parameter

[
\alpha
]

measures how strongly previous-state errors propagate through time.

### Case 1

[
\alpha<1
]

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

### Case 2

[
\alpha=1
]

Errors remain unchanged.

```text
100
 ↓
100
 ↓
100
```

Neutral stability.

---

### Case 3

[
\alpha>1
]

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

[
\boxed{\alpha<1}
]

is the key condition for long-term stability.

---

# Physical Meaning of β

The parameter

[
\beta
]

measures the sensitivity of the model to input perturbations.

A smaller value of (\beta) means:

* stronger robustness;
* better resistance to noise;
* smoother prediction behavior.

---

# Multi-step Stability

The theorem further proves

[
|\Delta u(T)|
\le
\alpha^T
|\Delta u(0)|
+
\frac{1-\alpha^T}{1-\alpha}
\beta
\sup_t|\Delta x(t)|.
]

This result implies:

1. Initial-state errors gradually vanish.
2. Input perturbations remain bounded.
3. Hidden-state trajectories remain stable over long horizons.

---

# Why Compensation Does Not Destroy Stability

The compensation mechanism contributes additional terms to both

[
\alpha
]

and

[
\beta.
]

However:

* all compensation gates are bounded;
* sigmoid-based gates have small derivatives;
* compensation signals are constrained.

Therefore, the compensation pathway introduces only limited perturbation amplification.

Instead, it helps prevent excessive information loss and improves the smoothness of hidden-state evolution.

---

# Overall Theoretical Conclusion

The theoretical analysis establishes three important properties of CBLSTM-SNP:

1. **Dynamic Balance**

   Hidden-state evolution is governed by the interaction of memory retention, spike consumption, and adaptive compensation.

2. **State Boundedness**

   Hidden states remain uniformly bounded and cannot diverge during long-term recurrent computation.

3. **Lipschitz Stability**

   Small perturbations in inputs lead to bounded changes in hidden states, ensuring robust and stable temporal modeling.

Together, these results provide theoretical support for the effectiveness and reliability of the proposed compensation-balanced recurrent architecture.
