# Tenth local test: stability-screened projected-gradient centre selection

This update adds a general, Toolbox-free point optimiser for the spherical-monogenic kernel from the previous milestone. You choose the ambient dimension \(m\), degree \(k\), and number of centres \(r\). It works in moderate dimension, subject to the cost of the full left-regular Gram eigensystem.

It is deliberately honest about what it does: projected gradient descent minimizes the smooth off-diagonal energy \(F\), then the exact Gram spectrum screens the endpoint for stability. It returns the best sampled local candidate; it does not prove a global minimum.

## New files

- src/clifford_spherical_monogenic_objective.m
- src/clifford_spherical_monogenic_optimize_centres.m
- tests/test_clifford_spherical_monogenic_optimizer.m
- run_spherical_monogenic_optimizer_test.m

## Required earlier modules

This update requires the previously installed kernel/Gram milestone and the left-regular eigensystem utilities:

- clifford_algebra.m
- clifford_spherical_monogenic_kernel.m
- clifford_spherical_monogenic_gram.m
- clifford_matrix_left_regular_matrix.m
- clifford_matrix_left_regular_eigensystem.m

No Optimization Toolbox or Symbolic Math Toolbox is required.

## The optimized energy and its exact sphere gradient

For unit points \(x,y\in S^{m-1}\), let \(t=\langle x,y\rangle\) and write

\[
K_{m,k}(x,y)=a_{m,k}(t)+(x\wedge y)b_{m,k}(t).
\]

For \(m>2\),

\[
a_{m,k}(t)=\frac{k+m-2}{m-2}C_k^{(m-2)/2}(t),
\qquad
b_{m,k}(t)=C_{k-1}^{m/2}(t),
\]

where \(b_{m,0}=0\). The coefficient energy of one off-diagonal kernel entry is

\[
q_{m,k}(t)=\left|K_{m,k}(x,y)\right|_{\mathrm{Cl}}^2
=a_{m,k}(t)^2+(1-t^2)b_{m,k}(t)^2.
\]

For rows \(\eta_i\) of \(X\), the objective is therefore

\[
F(X)=\sum_{i<j}q_{m,k}(\langle\eta_i,\eta_j\rangle).
\]

The objective utility evaluates this scalar formula without constructing a Clifford tensor at every gradient step. For \(m>2\) and \(k\geq1\),

\[
a'_{m,k}(t)=(k+m-2)b_{m,k}(t),
\]

\[
b'_{m,k}(t)=
\begin{cases}
0,&k=1,\\
mC_{k-2}^{(m+2)/2}(t),&k\geq2,
\end{cases}
\]

and

\[
q'_{m,k}(t)=2aa'-2tb^2+2(1-t^2)bb'.
\]

Thus the Riemannian gradient in row \(i\) is

\[
\operatorname{grad}_{\eta_i}F
=\sum_{j\ne i}q'_{m,k}(t_{ij})
\bigl(\eta_j-t_{ij}\eta_i\bigr).
\]

The code takes a negative tangent-gradient step, normalizes each row back to the sphere, and accepts it only if Armijo backtracking gives sufficient decrease in \(F\). The first row is fixed by default to remove global orthogonal-rotation symmetry; this does not exclude any Gram configuration up to an \(O(m)\) rotation.

For \(m=2\), the Chebyshev identity makes \(q_{2,k}(t)=1\), so \(F=\binom r2\) is exactly flat. Degree zero is also exactly flat in every dimension.

## Main function

~~~matlab
result = clifford_spherical_monogenic_optimize_centres(m, k, r, options);
~~~

For example, this asks for six degree-two centres on \(S^3\), makes the restarts reproducible, and requests a relative whitening-amplification cap of \(2\):

~~~matlab
options = struct( ...
    'numRestarts', 16, ...
    'maxIterations', 800, ...
    'randomSeed', 20261009, ...
    'relativeAmplificationLimit', 2);

result = clifford_spherical_monogenic_optimize_centres(4, 2, 6, options);
disp(result.summary)

if isempty(result.selected)
    disp('No sampled restart met the requested stability margin.');
    disp(result.bestStabilityRun)
else
    points = result.selected.points
    F = result.selected.objectiveF
    lambdaMin = result.selected.lambdaMin
    kappa2 = result.selected.conditionNumber2
end
~~~

The packed complex Gram matrix has dimension \(r\,2^m\). The default maximumPackedDimension = 1024 prevents an accidental oversized eigensystem. Increase it only deliberately, when memory and eigensolver time are available.

## Choosing epsilon

The fixed diagonal level is

\[
d=d_{m,k}=\binom{k+m-2}{k}.
\]

For a numerically positive-definite Gram matrix, the reported relative whitening amplification is

\[
A_{\mathrm{rel}}=\sqrt{\frac{d}{\lambda_{\min}}}.
\]

It equals \(1\) for the ideal scaled identity \(G=dI\). Choose the largest amplification \(A_{\max}\geq1\) that your reconstruction/noise budget permits, and let the code convert it to

\[
\epsilon=\frac{d}{A_{\max}^2}.
\]

This makes relativeAmplificationLimit easier to choose than a raw epsilon. For the earlier \(m=4,k=2\) example, \(d=6\):

| Desired \(A_{\max}\) | \(\epsilon/d\) | \(\epsilon\) |
|---:|---:|---:|
| \(3\) | \(0.1111\) | \(0.667\) |
| \(2\) | \(0.25\) | \(1.50\) |
| \(1.9365\) | \(0.2667\) | \(1.60\) |
| \(1.5\) | \(0.4444\) | \(2.667\) |

There is no universal best epsilon. It expresses the stability margin appropriate to the application. A condition-number target is different: always inspect kappa2 = lambdaMax/lambdaMin too.

- epsilon = [] means optimize \(F\) with no spectral target.
- epsilon = 0 is a PSD baseline, not a positive-definite demand.
- A positive epsilon is accepted only when the Gram matrix is numerically positive definite and lambdaMin >= epsilon.
- feasibilityTolerance is a small numerical comparison allowance used when selecting an endpoint near the requested threshold. The result separately reports isEpsilonAchieved (strict) and isEpsilonAchievedWithinTolerance.

If the physical design margin is \(\epsilon_{\mathrm{design}}\), use a slightly larger internal target or verify afterward that lambdaMin exceeds \(\epsilon_{\mathrm{design}}\) by an application-specific buffer.

## Necessary rank check

With \(N=2^m\), the packed Gram matrix is \((rN)\)-by-\((rN)\), while

\[
\operatorname{rank}\Lambda(G)\leq Nd_{m,k}.
\]

Hence a positive spectral margin is impossible if

\[
r>d_{m,k}.
\]

The function rejects that case before searching whenever epsilon is positive. The square interpolation choice \(r=d_{m,k}\) is natural, but it does not make every point configuration positive definite. In particular, \(d_{2,k}=1\), so more than one centre on the circle cannot meet a positive target.

## Why minimizing F alone is not enough

This repository uses the full left-regular map \(\Lambda\), not the smaller recursive \(\rho\) map used in the four-dimensional paper. Its exact identity is

\[
\left\|\Lambda(G)-dI\right\|_F^2=2NF.
\]

So a smaller \(F\) is valuable: it lowers the average off-diagonal energy. But it does not determine how that energy is distributed across spectral directions. A configuration with slightly higher \(F\) may have much larger \(\lambda_{\min}\).

That is precisely what your earlier \(m=4,k=2,r=6\) scan displayed: the lowest observed \(F\) was about \(40.231\), with \(\lambda_{\min}\approx0.262\); a slightly higher value \(F\approx40.533\) attained \(\lambda_{\min}\approx1.60\). The optimizer therefore:

1. minimizes smooth \(F\) from multiple starts;
2. computes the exact full Gram spectrum at every endpoint;
3. with an epsilon target, selects the lowest-\(F\) endpoint meeting the spectral requirement;
4. if no sampled endpoint meets it, leaves result.selected empty and returns result.bestStabilityRun for diagnosis.

The conservative sufficient certificate

\[
\lambda_{\min}(\Lambda(G))\geq d-\sqrt{2NF}
\]

is reported as frobeniusCertificateLowerBound. Equivalently,

\[
F\leq\frac{(d-\epsilon)^2}{2N}
\]

certifies a requested margin. It is sufficient only, not necessary, and is usually far too conservative for choosing epsilon in the six-centre \(m=4,k=2\) problem.

Do not compare raw \(F\) values across different \(m\), \(k\), or \(r\). Also do not compare raw Frobenius norms or eigenvalue multiplicities between this full-\(\Lambda\) representation and the paper's compact-\(\rho\) representation; their factors differ.

## Reading the result

result.summary has one row per restart with \(F\), lambdaMin, lambdaMax, kappa2, relative amplification, the conservative certificate, gradient norm, and strict/tolerance-aware feasibility fields.

When successful, result.selected includes the selected points, the coefficient Gram tensor, and the full eigensystem. result.globalMinimumProved is always false. Product-sphere nonconvexity and antipodal, permutation, rotation, and eigenvalue-multiplicity symmetries mean equivalent local minima are expected. Repeated values from many starts are useful evidence, not a global-minimum theorem.

## Merge and test

This archive has no enclosing clifford-adjoint-matlab folder. Extract it into the root of your repository. When Finder asks about src and tests, choose **Merge**, never Replace.

Open run_spherical_monogenic_optimizer_test.m in MATLAB and press Run. Expected output:

~~~text
Spherical-monogenic optimizer: objective, tangent gradient, stability selection, and edge cases passed.
~~~

The test checks the objective against the Gram builder, tangent-gradient finite differences, orthogonal invariance, a known two-centre \(m=4,k=1\) global oracle, epsilon conversion, rank rejection, constant-objective cases, option validation, and reproducibility.

