# CBLSTM-SNP
Theoretical Derivation
# Theoretical Interpretation of CBLSTM-SNP

## 1. Dynamic Interpretation of Compensation Mechanism
The recurrent update equation of CBLSTM-SNP is
      u(t) = r(t) ⊙ u(t−1) − c(t) ⊙ a(t) + k(t) ⊙ z(t)
This state evolution can be interpreted as the interaction of three different components:
## Memory Retention
      r(t) ⊙ u(t−1)
This term preserves historical information.The retention gate r(t) determines how much information from the previous state should be carried forward.
r(t) ≈ 1 → most historical information is preserved.r(t) ≈ 0 → historical information is forgotten.
This component plays a role similar to the forget gate in LSTM.
## Nonlinear Consumption
      c(t) ⊙ a(t)
This term performs spike-driven attenuation.The spike generation signal a(t) represents nonlinear neural activity.
The consumption gate c(t) controls how much of this activity should be used to reduce the current state.
Its main purpose is:
      (1)suppress redundant information
      (2)remove obsolete memory
      (3)prevent state accumulation
Without this term, the hidden state may continuously grow.
## Adaptive Compensation
      k(t) ⊙ z(t)
This is the most important component of CBLSTM-SNP.The compensation branch restores useful information that may have been excessively removed by nonlinear consumption.
The gate k(t) determines the compensation intensity, while z(t) provides the regulation signal.Unlike conventional recurrent networks, this branch explicitly introduces a correction mechanism.
Its role is:
      (1)prevent excessive state decay
      (2)smooth hidden-state evolution
      (3)improve long-term memory retention
      (4)enhance forecasting stability
## Dynamic Balance
The complete state evolution can be viewed as:
New State = Memory Retention − Nonlinear Consumption + Adaptive Compensation


## 2. Hidden State Boundedness
For recurrent neural networks, hidden states should remain finite.
If the state becomes extremely large:u(t) → ∞.The model becomes unstable.This is known as exploding states.
Therefore, it is necessary to prove that:u(t)always remains within a finite range.
The hidden state is updated by
      u(t) = r(t) ⊙ u(t−1) − c(t) ⊙ a(t) + k(t) ⊙ z(t)
      0 ≤ r(t), c(t), k(t), z(t) ≤ 1
      ||a(t)|| ≤ M_a
      
