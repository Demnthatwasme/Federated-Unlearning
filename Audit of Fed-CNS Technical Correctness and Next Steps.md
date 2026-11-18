c# Audit of Fed-CNS: Technical Correctness and Next Steps

*Review of FED\_CNS\_TECHNICAL\_DEEP\_DIVE.txt (Fed-CNS, contrastive null-space unlearning via an additive ledger). Based on the write-up only, not the code, so what the code actually does is unconfirmed. Some issues are weaknesses in the original FedNull design, not just the implementation.*

## Verdict

The linear algebra is right, but the numbers do not show that Fed-CNS unlearns anything in Setting B, and several claims in the write-up go further than the evidence.

## What's correct

- Subtracting additive statistics (A, S, n) is exact for those statistics.
- The Cholesky route to the generalized eigenproblem (`C = L⁻¹ Σ_f L⁻ᵀ`, `V = L⁻ᵀ u`) is right, and V comes out Σ\_r-orthonormal.
- The storage arithmetic for the ledger (1.02 MB) and the agreement fractions (410/423, 396/423) check out.

## What's wrong or unsupported (worst first)

### 1. In Setting B, Fed-CNS looks closer to the original model than to the retrained one

- D\_forget accuracy is 92.20%, against 92.43% for the original and 90.31% for retrain.
- D\_forget loss is 0.2954, against 0.3034 for the original and 0.3547 for retrain.
- The loss went down after "unlearning," so the model fits Client 09's data slightly better than before.
- FedEraser (89.36%, 0.4046) is closer to retrain on both metrics.
- The 96.9% agreement has no baseline. You need agreement(M\_orig, M\_retrain), which is probably already in the mid-90s since those models differ by about two accuracy points (a guess, so measure it). You also need agreement between two retrain seeds as a ceiling.
- 410 vs 396 is a 14-sample gap on a single seed.

### 2. The "private subspace" may not be private

- Dirichlet partitioning mostly changes class proportions, not feature distributions. So Σ\_f vs Σ\_r differs mainly because Client 09 has more bottles and beavers, which is label skew, not memorization.
- With n\_f = 423 < d = 512, Σ\_f is rank-deficient. Its top generalized eigenvalues are inflated by estimation noise, probably in directions where retained energy is tiny.
- **Control:** carve a random 423-sample pseudo-client out of the retained data and compare the eigenvalue spectra and the effect of the projection. If they look similar, the subspace is not capturing client-specific memorization.
- This applies to the original FedNull design too.

### 3. The head swap is probably doing most of the work

- Re-centering replaces the SGD-trained head with nearest-centroid prototypes.
- Run the pipeline with no client removed. If test accuracy falls from 93.0% to about 91.5% there, the drop comes from the classifier change, not unlearning.
- Also run recentering only, projection only, and projection onto a random subspace of the same k.
- In Setting A, `n_retained[bottle] = 0`, so the row is zeroed. That is the naive masking the brief calls cosmetic, so Setting A is not evidence for the projection either.
- A zeroed row gives a logit of 0, not −∞, and after projection cosines can go negative. Delete the row instead.

### 4. The Setting A table is mislabeled

- 80.10% and 80.70% are overall accuracy (0.9 × 89.00% and 89.67%), not retain accuracy.
- Fed-CNS's 79.40% implies about 88.2% retain accuracy, roughly 0.8 below retrain.
- Check how "100/100 agreement" is defined. If it means both models predict the same wrong class on every bottle, that is suspicious for FedEraser. If it only means both are "not bottle," it is trivial.

### 5. Projection details

- QR orthonormalization of V gives a Euclidean projector. The original design used the oblique `P = I − V Vᵀ Σ_r`, which is tied to retained energy. After QR nothing bounds the retained energy lost, so log `tr(QQᵀΣ_r)/tr(Σ_r)`.
- k is never stated. Choose it from a retained-energy budget.
- "P\_perp φ\_f = 0" holds only for φ\_f inside span(V\_k). That is not true of 423 samples in 512 dimensions with small k.
- A/n is an uncentered second moment, not a covariance. Post-ReLU features have a large shared mean.
- ρ = 1e-4 is absolute. Make it relative (`α·tr(Σ_r)/d`) and sweep it.
- Use float64 and symmetrize after subtracting, or Cholesky can fail.

### 6. Claims to soften

- **"The entire distribution is captured"** by A and S is wrong. They capture only first and second moments.
- **"100% offline"** hides two things. Retained clients were needed for the statistics round during training. And either client f must recompute A\_f, which needs its data, or the server stores per-client A\_k. Storing A\_k costs K MB, so storage is no longer independent of K. It is also a privacy leak unless secure aggregation protects it, which then blocks the subtraction.
- **"Exact"** holds only for the ledger. The statistics come from the final feature extractor, which still carries Client 09's influence in layer4. This is not exact unlearning and has no certificate.
- **Static extractor:** frozen BatchNorm does not make layer4 static, because layer4 is trained.
- **Storage:** the write-up's own arithmetic gives 67 × 42.63 = 2,856 MB, but it cites 2,200 MB. That figure is about right if only layer4 (\~8.4M parameters) is stored, so say so. Also pick 1.02 or 1.07 MB. Arithmetic is not a "proof."
- **Runtime:** the 0.51 s is mostly loading, with about 50 ms of math. `eigh` with eigenvectors is closer to 10⁹ FLOPs than 10⁸. Report server compute, client-f cost and training-time ledger cost separately. Baseline runtimes also drifted between runs (Setting A retrain 36.24 s → 27.63 s, FedEraser 11.70 s → 8.59 s), so repeat the runs.
- **"Virtually zero collateral degradation"** compares against FedEraser (91.6%). Against retrain (92.4%) and the original (93.0%), it is −0.9 and −1.5 points.
- **After deletion,** replace A\_all with A\_r. The stored statistics are personal data too.

## What to run next

1. M\_orig vs M\_retrain agreement, and retrain-seed vs retrain-seed agreement.
2. The "no client removed" control, plus the recentering-only, projection-only and random-subspace ablations.
3. The pseudo-client control.
4. 3–5 seeds with confidence intervals.
5. A membership-inference audit (a loss-threshold attack is enough to start). Fed-CNS's lower D\_forget loss will probably make members look more member-like, so move this ahead of the other milestones.
6. A sequential-request test: two deletions in a row.

## Improvements

- Contrast class-conditional second moments (a per-class ledger `A_{k,c}` costs about 10 MB) to remove the label-skew confound.
- Add a weight-level step on layer4, such as the FedNewton step, since the head-only approach cannot touch the memorized features.
- Present the method as a readout-scrubbing tier unless the controls above show the projection does real work.
