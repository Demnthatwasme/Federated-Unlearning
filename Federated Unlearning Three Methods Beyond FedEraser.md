
*Proposals based on the Federated Unlearning for Vision Models research brief (CIFAR-100, ResNet-18, Flower). None of the numbers below come from experiments; accuracy and storage figures are estimates to be tested on Settings A and B.*

## Shared primitive: an additive ledger instead of a history

Every method below uses the same trick. Second-moment statistics, gradients and Fisher diagonals are sums over samples, so the server never needs per-round or per-client history. After the last round, there is one extra "statistics round" in which each client k uploads, via secure aggregation:

- `A_k = Σ φ(x)φ(x)ᵀ` (512×512), and per-class feature sums `S_{k,c}` with counts `n_{k,c}`
- `g_k = Σ ∇ℓ_i(θ_T)` and `F_k = Σ (∇ℓ_i(θ_T))²` over layer4 and the head

The server keeps only the totals. Storage is O(|θ| + d²), independent of T and K. When client f leaves, the server computes `A_r = A_all − A_f`, `g_r = g_all − g_f`, and so on. Retained clients stay offline at unlearning time, and the statistics round is paid once during normal training.

Any stored per-client statistic derived from a client's data is itself personal data. Purge it on deletion, and replace the totals with the retained-only values afterward.

## Method 1: FedNull, contrastive null-space scrubbing (server-only)

**Formulation.** Let `Σ_f = A_f/n_f` and `Σ̃_r = A_r/n_r + ρI`. Find directions that are expensive in forget-client energy but cheap in retained energy:

`max_V tr(VᵀΣ_fV)  s.t.  VᵀΣ̃_rV = I_k`

This is the generalized eigenproblem `Σ_f v = λ Σ̃_r v`. Keep eigenvectors with `λ > τ`, and choose k so retained energy loss stays below η. The scrubbing map is the oblique projector `P = I − V Vᵀ Σ̃_r`, applied as `φ' = Pφ`. When `Σ̃_r ∝ I` this reduces to the brief's `W(I − P_forget)`.

Prototypes are then refreshed from retained data only: `μ_c^r = (S_c^all − S_c^f)/(n_c^all − n_c^f)`, `W_c = normalize(μ_c^r)`. If `n_c^r = 0`, as in Setting A, the row is deleted, which is exact.

**Workflow.**

- *Client f:* one forward pass over its data to compute `A_f, S_f`.
- *Server:* subtract, run the eigendecomposition, build P, refresh prototypes, and broadcast P (about 1 MB).
- *Retained clients:* nothing.

**Cost vs FedEraser.** Time is O(d³ + Cd) on the server, probably well under a second, against 11.7 s plus retained-client compute. Storage is about 1 MB against O(TK|θ|).

**Trade-offs.** It is the cheapest option and exact for the head in Setting A. The limit is that layer4 weights are untouched, so a white-box holder can bypass P. It is a read-out scrub, not weight erasure, and should be paired with Method 2.

## Method 2: FedNewton, one-shot influence removal from the ledger (server-only)

**Formulation.** Take one damped Newton step on the retained objective from the final model:

`θ_r ≈ θ_T − (Ĥ_r + λI)⁻¹ ∇L_r(θ_T)`

with `∇L_r = (g_all − g_f)/n_r` and `Ĥ_r = (F_all − F_f)/n_r`. The Fisher is a diagonal for layer4 and a full block for the head. Using `g_all` avoids assuming that 6 FL rounds reached a stationary point, which they likely did not. Add a trust region `‖Δθ‖ ≤ ρ` and tune λ and ρ offline against M\_retrain. Diagonal Fisher subtraction is exact, but K-FAC factors are not additive in this way, so avoid K-FAC here.

**Workflow.**

- *Client f:* recompute `g_f, F_f` at `θ_T`, using the same model the ledger was computed on.
- *Server:* subtract, compute the step, clip it, and apply it.
- *Retained clients:* none.

**Cost vs FedEraser.** Time is O(|θ|). Storage is about 2|θ|, roughly 67 MB for layer4 plus the head at fp32. With T=6 and K=10, FedEraser stores about 67|θ|, roughly 2 GB, so the saving is about TK/2 (30×).

**Trade-offs.** This should hit weights, not just the readout, so Grad-CAM drift on D\_forget should be real. Curvature weighting should limit collateral damage. The risks are the diagonal approximation and linearization error when the forgotten client holds about 10% of the data. Overshoot is the main failure mode: FedEraser's D\_forget loss of 0.4048 already exceeds the retrained model's 0.3547, so tune against retrain rather than maximizing forgetting.

## Method 3: FedCert, tiered certificates

**Tier 0 (exact).** If the backbone is frozen (ImageNet weights only, so data-independent), replace the cosine head with a ridge or softmax head on normalized features:

`W_r = (A_r + λI)⁻¹ B_r`, with `B_r = Σ φ yᵀ`

This equals retraining the head on the same features, so the unlearning is exact (ε = δ = 0) for the whole model, at O(d³ + d²C) time.

**Tier 1 (layer4 trainable).** Train layer4 with client-level clipped and noised updates (DP-FedAvg). The backbone is then (ε\_dp, δ) private per client, and the head is handled by Tier 0. Total ε follows by composition. Alternatively, use Guo et al.'s certified-removal recipe on a strongly convex head: a training-time perturbation `bᵀθ` with `b ~ N(0, σ_b²I)` (one seed, so O(1) storage) plus the Method 2 Newton step. The certificate is `ε = (ε'/σ_b)·√(2 ln(1.5/δ))`, where ε' bounds the gradient residual. The constants were written from memory, so check them against the paper. ε' grows roughly quadratically in `n_f/n_r`.

**Trade-offs.** With only 10 clients, client-level DP noise will likely cost real accuracy, and ε' is likely loose for a 423-image deletion. Run a privacy accountant before committing. No (ε, δ) certificate for a non-convex layer4 at this deletion size is within reach, and the paper should say so rather than claim one.

## Comparison

|  | Server-only | Retained clients | Storage | Server time | Guarantee |
| --- | --- | --- | --- | --- | --- |
| FedEraser | No | Calibrate | O(TK\|θ\|) | 11.7 s total | None |
| FedNull | Yes | None | \~1 MB | ms to sub-second | Exact for the head in Setting A |
| FedNewton | Yes | None | \~2\|θ\| | seconds | Empirical |
| FedCert | Yes | None | O(d² + \|θ\|) | seconds | Exact (Tier 0), composed (Tier 1) |

## Notes on the brief

1. **Citation.** FedEraser's first author is Gaoyang Liu (IWQoS 2021), not Wang. Please verify.
2. **">90% on CIFAR-100".** The results use classes 0–9, a 10-class problem. >90% on all 100 classes with this backbone and only layer4 trainable is not expected.
3. **Trajectory-drift claim.** The T>20 drift claim is untested, since all runs use 6 rounds. Run 20–30 rounds before citing it.
4. **MIA.** Certificate-style claims need the deferred MIA audit alongside Grad-CAM. Consider moving it up.
5. **Evaluation target.** Match M\_retrain (prediction agreement, loss gap, per-sample loss distribution) rather than maximizing forgetting.

**Suggested starting point:** FedNull plus FedNewton on Setting B, since those are cheapest to test against the 93.61% agreement baseline.
