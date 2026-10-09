# Eleventh local test: spherical-monogenic kernel-system design

This update adds the user-facing constructor

~~~matlab
design = clifford_spherical_monogenic_design(m, degree, options);
~~~

It chooses a set of centres on \(S^{m-1}\), refines them with the already
tested stability-screened sphere optimizer, and returns the selected
Clifford-valued Gram matrix and its full left-regular spectral diagnostics.

The result is a numerical system of kernel sections

\[
\bigl\{K_k(\,\cdot\,,\eta_j)\bigr\}_{j=1}^{r}.
\]

It is not yet a symbolic Appell/Jacobi listing of every spherical monogenic.
When \(r=d_{m,k}\) and the Gram matrix is numerically positive definite, it
is a numerically certified full kernel-section candidate at that degree.

## New files

- src/clifford_spherical_monogenic_design.m
- tests/test_clifford_spherical_monogenic_design.m
- run_spherical_monogenic_design_test.m

## Required earlier modules

Install this after the existing spherical-monogenic Gram and optimizer
milestones. In particular, it uses:

- clifford_algebra.m
- clifford_spherical_monogenic_objective.m
- clifford_spherical_monogenic_optimize_centres.m
- clifford_spherical_monogenic_gram.m
- clifford_matrix_left_regular_eigensystem.m

No Optimization Toolbox or Symbolic Math Toolbox is required.

## A recommended first design

For \(m=4\), degree \(k=2\), the diagonal level is

\[
d_{4,2}=\binom{4}{2}=6.
\]

Therefore the default creates six centres and a \(6\cdot2^4=96\)-by-\(96\)
complex left-regular Gram matrix.

~~~matlab
options = struct( ...
    'randomSeed', 20261009, ...
    'seedStrategy', 'hybrid', ...
    'greedyCandidateCount', 512, ...
    'numRestarts', 24, ...
    'relativeAmplificationLimit', 2);

design = clifford_spherical_monogenic_design(4, 2, options);

if design.isUsable
    centres = design.centres
    F = design.objectiveF
    lambdaMin = design.lambdaMin
    kappa2 = design.conditionNumber2
    relativeAmplification = design.relativeAmplification
else
    disp(design.status)
    disp(design.bestStabilityRun)
end
~~~

The option relativeAmplificationLimit \(=A_{\max}\) sets

\[
\epsilon=\frac{d_{m,k}}{A_{\max}^{2}}.
\]

Thus \(A_{\max}=2\) is usually easier to interpret than choosing a raw
epsilon: it asks for a relative whitening amplification no larger than
approximately \(2\).

## Why the default seed is hybrid

The constructor does not trust one random configuration, but it also does not
pretend that a greedy configuration is globally optimal.

With seedStrategy = hybrid:

1. The generated first centre is the canonical point \(e_1\).
2. At each later stage, the code samples greedyCandidateCount unit-sphere
   candidates.
3. It selects the candidate with smallest exact added kernel energy

   \[
   \Delta_j(x)
   =F(\eta_1,\ldots,\eta_j,x)-F(\eta_1,\ldots,\eta_j)
   =\sum_{i=1}^{j}
   \left|K_k(\eta_i,x)\right|_{\mathrm{Cl}}^2.
   \]

4. This kernel-aware seed becomes optimizer restart 1.
5. The remaining restarts are independent normalized Gaussian sphere points.
6. Every endpoint is then screened by the exact full Gram spectrum.

The greedy score is evaluated through the existing objective utility. It does
not duplicate the Gegenbauer formula, and its complete sampled candidate pool,
energies, and first-minimum choice are stored in design.seed.greedyTrace.

This is better suited to the problem than using a regular polyhedron as a
universal seed. A polyhedron maximizes ordinary Euclidean separation, whereas
the relevant pair energy depends on \(m\) and \(k\). For example, for
\(m=4,k=2\),

\[
q(t)=48t^4-16t^2+4,
\]

which is minimized at \(\lvert t\rvert=1/\sqrt6\), not at \(t=0\).
Moreover \(q(\pm1)=36=d^2\), so antipodal pairs in a cross-polytope are
energetically very bad despite being far apart in Euclidean distance.

For \(m=2\), or for degree zero in any dimension, the pair energy is constant.
The constructor then records a flat-objective fallback instead of falsely
claiming a meaningful greedy energy selection.

## Seed policies

| seedStrategy | Restart 1 | Later restarts |
|---|---|---|
| hybrid | kernel-aware greedy seed | random |
| random | random | random |
| kernel-greedy | kernel-aware greedy seed | none; numRestarts must be 1 |
| user | normalized initialPoints | random |
| continuation | normalized continuationPoints, optionally jittered | random |

Use user when you already know a configuration:

~~~matlab
options = struct( ...
    'pointCount', 3, ...
    'seedStrategy', 'user', ...
    'initialPoints', myPoints, ...
    'numRestarts', 8, ...
    'randomSeed', 20261009);

design = clifford_spherical_monogenic_design(4, 1, options);
~~~

Each row of initialPoints or continuationPoints is normalized, but its
orientation is otherwise preserved. The canonical \(e_1\) anchor is used only
for newly generated greedy seeds.

For a same-\(m\), same-degree warm start, use continuation:

~~~matlab
options = struct( ...
    'pointCount', size(oldDesign.centres, 1), ...
    'seedStrategy', 'continuation', ...
    'continuationPoints', oldDesign.centres, ...
    'continuationJitter', 0.01, ...
    'randomSeed', 20261009, ...
    'numRestarts', 8);
~~~

## Reproducibility

Set randomSeed to an integer from \(0\) through \(2^{32}-1\). The constructor
preserves MATLAB's caller RNG state. For a hybrid or kernel-greedy run with
master seed \(s\), it uses \(s\) for the greedy pool and \(s+1\), modulo
\(2^{32}\), for optimizer random restarts. Therefore changing the candidate
pool size does not silently change the independent restart sequence.

The returned design.seed structure records the requested seed, the derived
optimizer seed, the actual starting points, and the greedy trace.

## Important result fields

| Field | Meaning |
|---|---|
| design.centres | selected centres, or empty when no sampled endpoint met a requested positive target |
| design.gramEntries | Clifford-valued Gram tensor for selected centres |
| design.spectrum | full left-regular spectral diagnostics |
| design.optimizerResult | complete report from the previous optimizer |
| design.bestStabilityRun | best sampled run when a target fails |
| design.isEpsilonAchieved | requested positive epsilon is met strictly |
| design.isEpsilonAchievedWithinTolerance | selection met the tolerance-aware numerical screen |
| design.isNumericallyPositiveDefinite | full Gram matrix is numerically positive definite |
| design.isStableInterpolationCandidate | positive definite and, when requested, strictly meets epsilon |
| design.globalMinimumProved | always false in this numerical milestone |

The status field is one of:

- selected-no-epsilon-target
- selected-strictly-feasible
- selected-within-feasibility-tolerance
- selected-psd-baseline
- no-epsilon-feasible-candidate

In particular, a tolerance-only selection is never labelled as strictly
epsilon-certified. Lower \(F\) remains useful, but it is not by itself a
stability theorem or a proof of a global minimum.

## Options

The constructor-specific options are:

~~~matlab
pointCount = [];                 % default d_{m,k}
seedStrategy = 'hybrid';
greedyCandidateCount = 512;
initialPoints = [];              % required by 'user'
continuationPoints = [];         % required by 'continuation'
continuationJitter = 0;
randomSeed = [];
~~~

It also accepts the familiar optimizer options directly at the top level:

~~~matlab
epsilon
relativeAmplificationLimit
numRestarts
maxIterations
gradientTolerance
initialStepSize
armijoCoefficient
backtrackingFactor
minimumStepSize
feasibilityTolerance
maximumPackedDimension
anchorFirstPoint
verbose
~~~

The default packed-dimension guard is 1024. It protects against accidentally
forming an impractically large full left-regular eigensystem.

## Merge and test

This archive has no enclosing clifford-adjoint-matlab folder. Extract it into
the root of your repository. When Finder asks about src and tests, choose
**Merge**, never Replace.

In MATLAB, run:

~~~matlab
run_spherical_monogenic_design_test
~~~

Expected final line:

~~~text
Spherical-monogenic design: defaults, seed policies, Gram construction, and stability diagnostics passed.
~~~
