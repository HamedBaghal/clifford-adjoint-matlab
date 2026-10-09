# Ninth local test: general spherical-monogenic kernel and Gram matrix

This update adds the higher-dimensional spherical-monogenic reproducing kernel and turns a point ensemble on a sphere into the Clifford-valued Gram matrix needed for your centre-selection problem.

It is intentionally **not** the optimizer yet. First we need one trusted implementation of the kernel, its Gram matrix, the old objective `F`, and the spectral diagnostics. The next milestone can then optimize points without mixing algebra, normalization, and stability.

## New files

- `src/clifford_spherical_monogenic_kernel.m`
- `src/clifford_spherical_monogenic_gram.m`
- `tests/test_clifford_spherical_monogenic_gram.m`
- `run_spherical_monogenic_gram_test.m`

## Required earlier modules

The new Gram builder uses `clifford_algebra.m` and `clifford_multivector_star.m`. The test and practical workflow also use `clifford_matrix_left_regular_matrix.m`, `clifford_matrix_left_regular_eigensystem.m`, and `clifford_matrix_from_left_regular_matrix.m`.

## Kernel convention

Let \(m\geq3\), \(k\geq0\), \(\mu=(m-2)/2\), and \(x,y\in S^{m-1}\). The implemented convention is exactly

\[
K_{m,k}(x,y)
=\frac{k+m-2}{m-2}C_k^\mu(\langle x,y\rangle)
+(x\wedge y)C_{k-1}^{\mu+1}(\langle x,y\rangle),
\]

where the wedge term is zero for \(k=0\). For \(p<q\), the coefficient of \(e_pe_q\) is

\[
(x_p y_q-x_q y_p)C_{k-1}^{\mu+1}(\langle x,y\rangle).
\]

Gegenbauer values use a numeric three-term recurrence, so no Symbolic Math Toolbox is required.

### Dimension two

The formula has a removable singularity at \(m=2\). The code implements its exact limit:

\[
K_{2,0}(x,y)=1,
\qquad
K_{2,k}(x,y)=T_k(t)+(x\wedge y)U_{k-1}(t),\quad k\geq1,
\]

where \(t=\langle x,y\rangle\). Dimensions \(m=0\) and \(m=1\) are deliberately rejected by this sphere-kernel utility.

## Main function

```matlab
[G, kernelInfo] = clifford_spherical_monogenic_gram(points, k, alg)
```

Here `alg = clifford_algebra(m)` and `points` is an `r`-by-`m` real matrix: one point of \(S^{m-1}\) per row. The output `G` has size `r`-by-`r`-by-`N`, with \(N=2^m\), and is directly ready for the established block-map and eigensystem utilities.

By default every input row must already have norm one. To normalize nonzero approximate inputs explicitly:

```matlab
options = struct('normalizePoints', true);
[G, kernelInfo] = ...
    clifford_spherical_monogenic_gram(points, k, alg, options);
```

The output records original and used points, norm residuals, pairwise inner products, scalar and bivector kernel factors, and pairwise coefficient energies.

## Fixed diagonal and the number of points

For every \(x\in S^{m-1}\),

\[
K_{m,k}(x,x)=d_{m,k}\,1,
\qquad
d_{m,k}=\binom{k+m-2}{k}.
\]

This is both the diagonal level and the right-Clifford-module rank of degree-\(k\) spherical monogenics. The builder reports it as `kernelInfo.diagonalLevel` and `kernelInfo.monogenicModuleRank`.

Let \(N=2^m\).  The repository packs the Gram matrix with the full
left-regular representation, so

\[
\operatorname{rank}\Lambda(G)\leq N d_{m,k}.
\]

The packed matrix has size \(rN\)-by-\(rN\).  Consequently, with \(r\)
centres, positive definiteness is impossible if

\[
r>d_{m,k}.
\]

The natural square interpolation case is \(r=d_{m,k}\), but that alone does not guarantee positive definiteness: the actual configuration still matters.  In particular, \(d_{2,k}=1\) for every \(k\), so a fixed-degree two-dimensional ensemble with more than one centre can be positive semidefinite but cannot be positive definite.  In your earlier four-dimensional example,

\[
d_{4,2}=\binom{4}{2}=6,
\]

which exactly matches the diagonal value and six-centre choice.

## Hermitian structure and positivity

For real sphere points,

\[
K_{m,k}(x,y)^\star=K_{m,k}(y,x).
\]

The code evaluates the upper triangle and fills the lower triangle by Clifford star. Thus the tensor is star-self-adjoint by construction, and the full block left-regular matrix is Hermitian. Because this is a reproducing kernel, the exact block Gram matrix is positive semidefinite. It can still be singular—for repeated centres, poor configurations, or too many centres.

## `F` versus \(\lambda_{\min}\)

The builder returns

\[
\texttt{kernelInfo.objectiveF}
=\sum_{i<j}\lvert K_{m,k}(\eta_i,\eta_j)\rvert_{\mathrm{Cl}}^2.
\]

Every diagonal is the same scalar \(d_{m,k}\), hence

\[
\big\|\Lambda(G)-d_{m,k}I\big\|_F^2=2N F,
\qquad N=2^m.
\]

Here \(\Lambda\) is this repository's full \(N\)-by-\(N\) left-regular representation of each Clifford entry.  It is not the smaller recursive \(\rho\) representation used in the four-dimensional paper.  For example, at \(m=4\) this identity has factor \(2N=32\); a compact \(\rho\)-based calculation can have a different Frobenius factor (such as \(8F\)).  Do not compare raw Frobenius norms or eigenvalue multiplicities across those two representations.  In exact arithmetic, faithful \(*\)-representations describe the same spectral support and hence agree on the extremal spectral values of the same Hermitian algebra element.

So lowering `F` is useful: it makes the Gram matrix closer to a scaled identity in average Frobenius energy.  `F` is unnormalised, so compare it only among ensembles with the same \(m\), \(k\), and number of centres \(r\).  It is not the same as maximizing the worst spectral direction: two ensembles can have ordered values of `F` but the opposite ordering in \(\lambda_{\min}\).

The identity also gives the conservative one-way certificate

\[
\lambda_{\min}(\Lambda(G))
\geq d_{m,k}-\sqrt{2NF}.
\]

Thus \(F<d_{m,k}^2/(2N)\) is sufficient for positive definiteness.  The converse is false, so this is a certificate rather than a rule for selecting the best ensemble.

Use `F` as a smooth, inexpensive search objective, then evaluate \(\lambda_{\min}\) for candidates and select an ensemble only if it meets your design requirement \(\lambda_{\min}\geq\epsilon\). This \(\epsilon\) is a Gram-spectrum margin; it is unrelated to the floating-point `unitTolerance` used to validate sphere points.

## Obtaining \(\lambda_{\min}\) and the condition number

The builder does not run an eigensolver automatically, because an optimizer may evaluate `F` thousands of times. Use the existing tested utility when you need stability information:

```matlab
m = 4;
k = 2;
alg = clifford_algebra(m);

points = [1 0 0 0;
          0 1 0 0;
          0 0 1 0];

[G, kernelInfo] = clifford_spherical_monogenic_gram(points, k, alg);
spectrum = clifford_matrix_left_regular_eigensystem(G, alg);

lambdaMin = spectrum.hermitian.lambdaMin
lambdaMax = spectrum.hermitian.lambdaMax
kappa2 = spectrum.hermitian.positiveDefiniteConditionNumber2
F = kernelInfo.objectiveF
```

For a positive-definite Gram matrix, `lambdaMin` and `kappa2` are the correct stability diagnostics.

## Merge and test

This archive has no enclosing `clifford-adjoint-matlab` folder. Extract its contents into the root of your repository. When Finder asks about `src` and `tests`, choose **Merge**, never Replace.

Open `run_spherical_monogenic_gram_test.m` in MATLAB and press **Run**. Expected output is:

```text
Cl_{0,2}: spherical-monogenic kernel, Gram matrix, and block diagnostics passed.
Cl_{0,3}: spherical-monogenic kernel, Gram matrix, and block diagnostics passed.
Cl_{0,4}: spherical-monogenic kernel, Gram matrix, and block diagnostics passed.
Cl_{0,5}: spherical-monogenic kernel, Gram matrix, and block diagnostics passed.
```

The test independently checks degree zero, the universal degree-one formula, the exact \(m=2\) trigonometric limit, closed degree-two formulas in dimensions 3–5, star-Hermitian structure, positive semidefiniteness, singular Gram cases, the \(m=4,k=2\) objective formula, permutation covariance, block-map energy identities, decoder recovery, and invalid inputs.
