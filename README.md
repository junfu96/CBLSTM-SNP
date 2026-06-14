# CBLSTM-SNP
Theoretical Derivation
# The overall proof of the three-part theory
CBLSTM-SNP is not a model that will cause the state to become increasingly chaotic over time. Its hidden state is controllable and stable to input perturbations. Therefore, it is introduced in three layers:
1. The Dynamic Interpretation section explains what the model is "doing".
2. The Boundedness of Hidden State section proves that the state will not grow indefinitely.
3. The Multi-step Lipschitz Stability section proves that even a slight change in the input will not cause the output to become drastically out of control, and long-term propagation will not diverge.
## 1. Dynamic Interpretation
The updated formula: u(t) = r(t) ⊙ u(t−1) − c(t) ⊙ a(t) + k(t) ⊙ z(t) can be broken down.First, the first part, r(t) ⊙ u(t−1), represents memory retention. It indicates that not all of u(t−1) from the previous time step is discarded; a portion is retained. The larger r(t) is, the more memory is retained; the smaller the r(t) is, the more information is forgotten. Second, c(t) ⊙ a(t) indicates that the model actively consumes a portion of the state, removing redundant or outdated information. a(t) is the impulse generation term; c(t) is the consumption gate, controlling how much is removed. This helps with noise reduction and redundancy removal. Finally, k(t) ⊙ z(t) is adaptive compensation. Its function is to bring back some useful information when the previous two terms weaken the state too much. It has three functions: preventing excessively rapid state decay; preventing excessive oscillation of hidden states; and making state evolution smoother.
## 2. Boundedness of Hidden State
The core of the proof is that the hidden state u(t) will not increase infinitely over time.For example, there will be no explosive growth phenomenon like 1 → 2 → 5 → 20 → 100 → ∞.Starting from the update equation, we take the norm of both sides to observe the maximum possible threshold, regardless of the sign. This yields the following equation:

||u(t)|| ≤ ||r(t) ⊙ u(t−1)|| + ||c(t) ⊙ a(t)|| + ||k(t) ⊙ z(t)|| Since r(t) ranges from 0 to 1, the first term at most shrinks or preserves u(t-1), without significantly altering its value. Therefore:



||r(t) ⊙ u(t−1)|| ≤ ||r(t)||∞ · ||u(t−1)||



Let's further assume ||r(t)||∞ ≤ γ < 1, then this term becomes:

≤ γ · ||u(t−1)||

This shows that the influence of the previous state on the next state is "contractual," not "explosive." Similarly, the second term is also finite, not explosive.

In the third term, k(t) and z(t) are both bounded, so the third term can only contribute a finite amount and will not cause divergence.

## 3.Multi-step Lipschitz Stability
This section proves that if the input changes slightly, the hidden state will not be amplified excessively. We define the original perturbation and the perturbation trajectory. Then we define the difference:

Δu(t) = u(t) − ũ(t)


Δx(t) = x(t) − x̃(t)

In the following equation,
||Δu(t)|| ≤ α ||Δu(t−1)|| + β ||Δx(t)||
this means that α controls the propagation of "old errors"; β controls the propagation of "input disturbances".

The product difference is broken down as follows:
c ⊙ a − c̃ ⊙ ã = c ⊙ (a − ã) + (c − c̃) ⊙ ã


k ⊙ z − k̃ ⊙ z̃ = k ⊙ (z − z̃) + (k − k̃) ⊙ z̃

||Δu(t)|| ≤ α ||Δu(t−1)|| + β ||Δx(t)||
When α < 1, the error will not accumulate out of control, but will be controlled through geometric decay. After multi-step expansion, we get:
||Δu(T)|| ≤ α^T ||Δu(0)|| + [(1−α^T)/(1−α)] β sup ||Δx(t)||
This indicates that the initial error will become smaller and smaller; the impact of input disturbances is bounded; and long-term predictions will not diverge.
