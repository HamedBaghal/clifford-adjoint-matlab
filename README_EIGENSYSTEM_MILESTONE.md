# Fifth local test: eigenvalues, eigenvectors, and stability of one multivector

This update adds a spectrum utility on top of the faithful complex
left-regular representation. It requires the four earlier modules:

- clifford_algebra.m
- clifford_multivector_product.m
- clifford_multivector_star.m
- clifford_left_regular_matrix.m

## New files

- src/clifford_left_regular_eigensystem.m
- tests/test_clifford_left_regular_eigensystem.m
- run_eigensystem_test.m

## What this utility solves

For one element

\[
A\in\mathbb C\otimes \mathrm{Cl}_{0,m},
\]

the earlier map constructs its left-regular matrix

\[
L_A=\lambda_m(A),\qquad \lambda_m(A)X=AX.
\]

With S = clifford_left_regular_eigensystem(A,alg), the important outputs
are:

| Field | Meaning |
| --- | --- |
| S.eigenvalues | All ordinary complex eigenvalues of \(L_A\), including multiplicities. |
| S.rightEigenvectors(:,j) | Blade-coefficient vector of a multivector \(X_j\) satisfying \(A X_j=\lambda_j X_j\). |
| S.relativeEigenResiduals | Residual of each returned eigenpair. |
| S.sigmaMin, S.sigmaMax | Smallest and largest singular values of \(L_A\). |
| S.conditionNumber2 | Thresholded 2-norm conditioning diagnostic; Inf when the chosen numerical rank test says singular. |
| S.rawConditionNumber2 | The direct quotient \(\sigma_{\max}/\sigma_{\min}\), when the latter is nonzero. |
| S.hermitian | Extra information for the symmetrized element when \(A\) is numerically star-self-adjoint. |

This is deliberately a utility for one multivector. It does not yet accept
an \(r\times r\) matrix whose entries are multivectors. The next
matrix-level step will assemble the block matrix

\[
[\lambda_m(A_{ij})]_{i,j=1}^r
\]

and then use the same numerical diagnostics on that ordinary complex block
matrix.

## Why its eigenvalue values are the correct algebra spectrum

Let \(z\in\mathbb C\). Since the map is unital and complex-linear,

\[
zI-L_A=L_{z1-A}.
\]

Therefore

\[
zI-L_A\text{ is singular}
\quad\Longleftrightarrow\quad
z1-A\text{ is not invertible in }\mathbb C\otimes\mathrm{Cl}_{0,m}.
\]

The reverse implication is immediate from a multiplicative representation.
For the forward implication, if \(L_{z1-A}\) is invertible, solve
\((z1-A)Y=1\). Injectivity of the same left-multiplication map then gives
\(Y(z1-A)=1\), so \(z1-A\) is a unit. Thus the set of values returned by
eig(\(L_A\)) is the correct complex algebra spectrum.

What is not canonical is the displayed multiplicity and the individual
coefficient eigenvectors. They come from the regular module:

- If \(m=2r\), then
  \(\mathbb C\otimes\mathrm{Cl}_{0,m}\cong M_d(\mathbb C)\), with
  \(d=2^r\), and the regular matrix contains \(d\) copies of the
  irreducible spectrum.
- If \(m=2r+1\), then it is
  \(M_d(\mathbb C)\oplus M_d(\mathbb C)\). The regular matrix contains
  both branches, each with \(d\)-fold regular repetition.

So repeated values are expected even before any additional geometric
degeneracy occurs. The structure is recorded in S.regularRepresentation.
For odd \(m\), `regularCopyMultiplicity = d` refers to each simple branch;
coincident values from the two branches can have larger total multiplicity.

## Basic use

    alg = clifford_algebra(4);
    n = alg.nBlades;

    one = zeros(n, 1);  one(1) = 1;
    e1  = zeros(n, 1);  e1(2)  = 1;
    A = 2 * one + 1i * e1;

    S = clifford_left_regular_eigensystem(A, alg);
    S.eigenvalues
    S.rightEigenvectors(:, 1)
    S.sigmaMin
    S.conditionNumber2

Each column of S.rightEigenvectors is a multivector written in the
package's blade-coefficient basis. It is normal for a nonnormal or repeated
problem for these columns to be nonunique or poorly conditioned. Use
S.eigenvectorMatrixConditionNumber2 as a warning diagnostic; never compare
the entries of two eigenvectors directly, because phases and bases in a
repeated eigenspace are arbitrary. This diagnostic can be very large or Inf
at a repeated or defective eigenvalue; it is not an invertibility test.

## Hermitian, positive-definite case

The Clifford adjoint is implemented by clifford_multivector_star. If

\[
A=A^\star,
\]

then \(L_A=L_A^H\). The utility recognizes this up to its stated numerical
tolerance, forms \(H=(L_A+L_A^H)/2\), and places a sorted real eigensystem
in S.hermitian.

If the input passed only the tolerance test rather than being exactly
self-star, every object in S.hermitian belongs to this symmetrized matrix
\(H\), equivalently to the symmetrized multivector
\(h=(A+A^\star)/2\).  Its coefficient vector is
S.hermitian.symmetrizedCoefficientVector.  The raw fields directly under S
always remain the eigensystem and diagnostics of the original \(L_A\).

    b = randn(n, 1) + 1i * randn(n, 1);
    Bstar = clifford_multivector_star(b, alg);
    H = clifford_multivector_product(Bstar, b, alg);
    H(1) = H(1) + 2;       % H = B^star B + 2*1 > 0

    options = struct('computeSpectralProjectors', true, ...
                     'computeInverseSquareRoot', true);
    S = clifford_left_regular_eigensystem(H, alg, options);

    S.hermitian.lambdaMin
    S.hermitian.lambdaMax
    S.hermitian.positiveDefiniteConditionNumber2
    W = S.hermitian.inverseSquareRoot;

For this branch, \(\lambda_{\min}>0\) is meaningful and
positiveDefiniteConditionNumber2 is
\(\lambda_{\max}/\lambda_{\min}\). The inverse square root is produced only
when lambdaMin > positiveDefiniteTolerance; its residual checks
\(W H W\approx I\). When numerical reconstruction confirms that it lies
in the left-regular image, its coefficient vector is also returned as
inverseSquareRootCoefficientVector and
inverseSquareRootIsInLeftRegularImage is true.

The optional projectors are complete numerical eigenvalue-cluster
projectors, not arbitrary rank-one projectors from individual MATLAB
eigenvectors. Their matrix versions are always returned on the Hermitian
branch. A coefficient vector is returned only when direct reconstruction
confirms that the projector lies in the left-regular image; check the
corresponding spectralProjectorIsInLeftRegularImage entry before using it.

For a non-Hermitian element there is no ordered generic “smallest
eigenvalue.” In that case use sigmaMin, not min(abs(S.eigenvalues)), to
measure distance from singularity and possible linear-solve amplification.

## Choosing the numerical tolerance

There is no universal best numerical epsilon. The default has a clear role:

\[
\tau_{\rm rel}=100N\,\varepsilon_{\rm mach},\qquad N=2^m,
\]

and value-scale thresholds are

\[
\tau=\tau_{\rm abs}+\tau_{\rm rel}\,\sigma_{\max}(L_A).
\]

The defaults are relativeTolerance = 100*N*eps and absoluteTolerance = 0.
This avoids incorrectly declaring a well-scaled but tiny invertible
multivector singular merely because its coefficients are small. If your
input coefficients have known relative uncertainty \(\delta_{\rm rel}\), a
sensible choice is

    options.relativeTolerance = max(100 * alg.nBlades * eps, deltaRel);
    options.absoluteTolerance = deltaAbs;

The available overrides are:

| Option | Role |
| --- | --- |
| relativeTolerance, absoluteTolerance | Base numerical resolution. |
| singularValueTolerance | Explicit cutoff for numerical rank and conditionNumber2. |
| eigenvalueClusterTolerance | Maximum diameter of a greedy numerical cluster; it only groups raw values for reporting and never removes them. |
| selfAdjointTolerance, normalityTolerance | Dimensionless residual tests. |
| positiveDefiniteTolerance | Threshold used to decide whether \(\lambda_{\min}\) is safely positive. |

The left-regular image checks for optional projectors and inverse square roots
use the relative threshold S.hermitian.leftRegularImageTolerance, equal to
max(relativeTolerance, 100*N*eps).  This is intentionally scale-free because
the reported image residuals are relative Frobenius residuals.

Do not use eigenvalueClusterTolerance to force a desired multiplicity. For a
nonnormal or defective matrix, small perturbations can split repeated
eigenvalues by much more than machine epsilon. Cluster output is a numerical
diagnostic, not a proof of exact multiplicity. Each reported cluster has
cluster.diameter no greater than its requested tolerance; grouping can still
depend on the ordering of values near the boundary, so inspect the raw values
as well.

Also do not confuse these numerical tolerances with a design constraint such
as \(\lambda_{\min}\ge\epsilon_{\rm design}\) in an optimization. The latter
is a problem-dependent stability target; it should be chosen from the
permitted amplification or condition number, not from machine epsilon.

## Why singular values are included

For \(n=e_1+i e_2\), one has \(n^2=0\). Hence \(1+n\) is invertible with
inverse \(1-n\), and every eigenvalue of its left-regular matrix is \(1\).
Nevertheless it is nonnormal and has \(\sigma_{\min}<1\). Thus a spectrum
that looks safely away from zero can still have poor inversion stability.
sigmaMin and the condition number detect this; eigenvalues alone do not.

## Merge and test

This archive has no enclosing clifford-adjoint-matlab folder. Copy or unzip
its contents into the root of your existing repository. When Finder asks
about src and tests, choose Merge, never Replace.

Open run_eigensystem_test.m in MATLAB and press Run. Expected output:

    Cl_{0,4}: 16-by-16 eigensystem; spectral diagnostics passed.
    Cl_{0,5}: 32-by-32 eigensystem; spectral diagnostics passed.

The test verifies generic eigenpair residuals both as complex matrix equations
and Clifford-product equations; edge dimensions; a nonzero zero divisor; a
nonnormal invertible element; self-star positive-definite spectra; optional
projectors and inverse square roots; scalar clustering; and invalid options.
