# NCA R1 Reproducibility Record

This directory contains the compact reproducibility record for the
Neural Computing and Applications R1 revision of LUT-KAN.

Large intermediate artifacts, generated LUT files, resolved run
configurations, and temporary benchmark outputs are intentionally not tracked
in Git. The files retained here are sufficient to identify the experimental
protocols, aggregate the reported results, and audit the revised numerical
claims.

## Repository state

The revision is based on the endpoint-consistent LUT implementation introduced
on the `fix/ncaa-r1-endpoint-grid` branch.

The revised LUT contract uses endpoint-inclusive sampling within every spline
segment:

    x[k,l] = t[k] + l/(L-1) * (t[k+1] - t[k]),
    l = 0, ..., L-1.

Thus, `L` denotes the number of stored LUT samples per segment, and linear
interpolation uses the corresponding `(L-1)` interval scale.

The sampling geometry is independent of the runtime boundary mode.
`closed` and `half_open` specify boundary-membership/OOB semantics rather
than different LUT construction grids.

For the quantized LUTs used in these experiments, affine quantization
parameters are derived directly from the sampled LUT values of each segment.
No external calibration dataset is used to determine the quantization ranges.

## 1. Controlled numerical study

Directory:

`controlled/`

This is the controlled matched-backend experiment used to evaluate numerical
error, memory, and latency as a function of LUT resolution.

Primary settings include:

- PyKAN B-spline layer
- architecture: 10 -> 8
- grid: 8
- spline order: 3
- L in {16, 32, 64, 128}
- symmetric int8 and asymmetric uint8 LUTs
- five random seeds
- endpoint-inclusive LUT construction
- linear interpolation
- closed boundary mode
- clip-x OOB policy for the main comparison

The full per-run records are in `all_runs.csv`, and aggregate tables are
provided separately for accuracy, latency, and memory.

### Interpretation

The corrected endpoint-inclusive implementation does not support the previous
empirical O(1/L) approximation-error claim.

Instead, increasing L reduces interpolation error initially, after which the
8-bit quantization error becomes dominant and the observed error reaches a
quantization floor.

In the controlled study, L=32 provides a conservative numerical
accuracy-memory operating point. Increasing L further yields little additional
accuracy while approximately doubling LUT storage at each step.

The controlled numerical operating point should not be interpreted as a
universal task-level optimum; the downstream DoS experiment below demonstrates
that a smaller LUT can already saturate application-level performance.

## 2. OOB stress study

Directory:

`oob_stress/`

This experiment evaluates LUT behavior under genuine out-of-domain inputs.

Unlike the main controlled experiment, the Gaussian evaluation inputs are not
pre-clipped to the LUT domain. For the 10-dimensional N(0,1) input distribution
and the LUT domain [-1.75, 1.75], approximately 56.5% of evaluated input vectors
contain at least one out-of-domain coordinate.

The stress test covers:

- closed and half-open boundary semantics;
- clip-x and zero-spline OOB policies;
- L=64;
- symmetric int8 LUTs;
- five random seeds.

The results show that the compiled LUT evaluator reproduces the corresponding
policy-defined reference behavior with numerical errors comparable to the
in-domain case.

For continuous inputs, closed and half-open boundary conventions yield nearly
identical OOB rates because the probability of sampling exactly the upper
endpoint is negligible.

The comparison is a test of semantic fidelity. It does not establish that
either clip-x or zero-spline is an adversarial defense mechanism.

## 3. CICIDS2017 DoS case study: primary batch-1 benchmark

Directory:

`dos_b1/`

This is the primary latency-critical downstream deployment benchmark used for
the revised paper.

The trained KAN model and leakage-free CICIDS2017 preprocessing pipeline are
provided by:

`KuznetsovKarazin/kan-dos-detection`

Frozen R1 tag:

`ncaa-r1-dos-20260815`

The binary case study uses:

- 231,073 BENIGN flows;
- 231,073 DoS Hulk flows;
- 369,716 training flows;
- 92,430 held-out test flows;
- 78 input features;
- KAN architecture 78 -> 32 -> 16 -> 1;
- grid=5;
- spline order k=3;
- threshold=0.5.

The revised LUT evaluation uses the same trained KAN for all L values and
compares matched implementations of:

- Float PyKAN;
- NumPy B-spline;
- NumPy LUT;
- Numba B-spline;
- Numba LUT.

LUT resolutions:

- L=16
- L=32
- L=64
- L=128

### Classification result

Float PyKAN:

- Accuracy: 0.98995997
- Precision: 0.98438402
- Recall: 0.99571568
- F1: 0.99001743
- ROC-AUC: 0.99910626
- PR-AUC: 0.99905396

For L=64 and L=128, the LUT confusion matrix matches Float PyKAN exactly.

For L=16 and L=32, the LUT result differs from Float PyKAN by only one benign
test flow. This small numerical difference is not interpreted as an accuracy
improvement; application-level classification performance is effectively
unchanged.

### Accuracy-latency-memory Pareto

For this specific downstream classifier, L=16 is the deployment-oriented
operating point:

- task-level performance is already saturated;
- LUT storage is approximately 0.669 MiB;
- median batch-1 Numba LUT latency is approximately 0.028 ms;
- median matched Numba B-spline -> LUT speedup is approximately 26.8x.

Increasing L from 16 to 128 increases LUT storage by more than sixfold without
a measurable classification benefit.

This task-level result complements, rather than contradicts, the controlled
study: L=32 is a conservative generic numerical operating point, whereas this
specific classifier reaches task-level saturation already at L=16.

The primary latency claim is based on matched Numba B-spline and Numba LUT
implementations, not on a cross-framework comparison against PyTorch/PyKAN.

## 4. Supplementary batch-scaling benchmark

Directory:

`dos_batch_scaling/`

This directory retains the multi-batch throughput experiment for batch sizes:

- 1
- 16
- 256
- 1024

It is supplementary evidence for scaling behavior.

The primary deployment result in the revised manuscript is the batch-1
matched-backend benchmark in `dos_b1/`.

## Validation

`dos_b1/validation_checks.csv` verifies numerical agreement between:

- Float PyKAN and the matched NumPy B-spline implementation;
- NumPy and Numba B-spline implementations;
- NumPy and Numba LUT implementations at each L.

All registered validation checks pass the specified numerical tolerances.

## Archived raw results

Complete local experiment bundles used to prepare this record were archived
separately before repository finalization. They are not committed to Git to
avoid tracking generated artifacts and large benchmark directories.

## Relation to the revised manuscript

The main methodological corrections represented by this record are:

1. endpoint-consistent LUT sampling and interpolation;
2. explicit separation of LUT sampling geometry and boundary semantics;
3. clarification that quantization ranges are derived from sampled LUT values,
   not from an external calibration dataset;
4. a genuine OOB stress test;
5. matched DoS-specific NumPy/Numba B-spline baselines;
6. a downstream accuracy-latency-memory Pareto analysis over L;
7. use of the leakage-free CICIDS2017 model from
   `kan-dos-detection@ncaa-r1-dos-20260815`.

These artifacts are intended to make the numerical claims in the R1 revision
independently auditable.
