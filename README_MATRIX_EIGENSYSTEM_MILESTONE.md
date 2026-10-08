# Seventh local test: Clifford-matrix eigensystem and stability diagnostics

This update solves the eigenvalue and conditioning problem for a square
matrix whose entries are complex Clifford multivectors. It builds on the
already-tested full block left-regular map.

It requires these earlier source files:

- `clifford_algebra.m`
- `clifford_multivector_product.m`
- `clifford_multivector_star.m`
- `clifford_left_regular_matrix.m`
- `clifford_matrix_left_regular_matrix.m`

## New files

- `src/clifford_matrix_left_regular_eigensystem.m`
- `tests/test_clifford_matrix_left_regular_eigensystem.m`
- `run_matrix_eigensystem_test.m`

## What problem it solves

Let

\[
R=\mathbb C\otimes \mathrm{Cl}_{0,m},\qquad N=2^m,
\qquad A=[a_{ij}]\in M_r(R).
\]

The earlier block-map function constructs

\[
\Lambda_r(A)=\bigl[\lambda_m(a_{ij})\bigr]_{i,j=1}^r
\in M_{rN}(\mathbb C),
\]

where each \(\lambda_m(a_{ij})\) is the \(N\times N\) left-regular matrix
of one Clifford multivector. The new function is

    S = clifford_matrix_left_regular_eigensystem(entries, alg);

and computes the ordinary complex eigensystem of \(\Lambda_r(A)\).

If \(v_j\) is column `j` of `S.packedRightEigenvectors` and
\(\zeta_j=\texttt{S.eigenvalues(j)}\), then \(v_j\) encodes an
\(r\)-tuple of Clifford multivectors

\[
X_j=(x_1,\ldots,x_r)^T,
\]

such that

\[
A X_j=\zeta_j X_j,
\]

with \(\zeta_j\in\mathbb C\) central. The packing is

\[
v_j=
\begin{bmatrix}
\operatorname{coeff}(x_1)\\
\vdots\\
\operatorname{coeff}(x_r)
\end{bmatrix}.
\]

This is deliberately the full packed **left-action / central-complex**
eigenproblem. It does
not claim to solve a noncentral or right-Clifford eigenvalue problem.

## Why the returned eigenvalue values are correct

For every \(z\in\mathbb C\),

\[
zI_{rN}-\Lambda_r(A)=\Lambda_r(zI_r-A).
\]

The block map is a faithful unital algebra homomorphism. Therefore

\[
zI_r-A\text{ is invertible in }M_r(R)
\quad\Longleftrightarrow\quad
zI_{rN}-\Lambda_r(A)\text{ is nonsingular}.
\]

For the reverse implication, if the complex matrix on the right is
invertible, its inverse is a polynomial in that matrix by finite-dimensional
linear algebra. Because the image of \(\Lambda_r\) is a unital subalgebra,
that polynomial is again the image of an inverse of \(zI_r-A\).

Thus the values in `S.eigenvalues` are exactly the **central complex
spectrum** of \(A\). Their raw multiplicities and the displayed eigenvectors
are not intrinsic: they depend on this block left-regular representation.

## Basic use

The canonical numeric input is the same as the preceding block-map module:

\[
\texttt{entries(i,j,k)}
=\text{coefficient of the blade with mask }k-1\text{ in }a_{ij}.
\]

For example, here is a \(2\times2\) matrix in \(\mathrm{Cl}_{0,4}\):

    alg = clifford_algebra(4);
    N = alg.nBlades;

    one = zeros(N, 1);  one(1) = 1;
    e1  = zeros(N, 1);  e1(2)  = 1;
    e2  = zeros(N, 1);  e2(3)  = 1;

    A = zeros(2, 2, N);
    A(1,1,:) = reshape(2 * one + 1i * e1, [1, 1, N]);
    A(1,2,:) = reshape(e2, [1, 1, N]);
    A(2,1,:) = reshape(-e1, [1, 1, N]);
    A(2,2,:) = reshape(one, [1, 1, N]);

    S = clifford_matrix_left_regular_eigensystem(A, alg);
    S.eigenvalues
    S.sigmaMin
    S.conditionNumber2

You may instead pass an `r-by-r` cell matrix whose cells are `N-by-1`
coefficient vectors. At \(m=0\), an ordinary complex `r-by-r` MATLAB matrix
is accepted directly.

For the \(2\times2\), \(m=4\) example, `S.blockLeftRegularMatrix` is
`32-by-32`.

## Reading the eigenvectors

The primary field is the ordinary complex matrix

    V = S.packedRightEigenvectors;

Its column `j` is a packed Clifford column. The same information is supplied
in a MATLAB-friendly array:

    C = S.rightEigenvectorCoefficients;
    x_i = C(i, :, j).';

Here `x_i` is the `N-by-1` coefficient vector of component \(i\) of the
\(j\)-th eigenvector. The array has size

\[
r\times N\times(rN),
\]

and `C(i,k,j)` is the coefficient of blade mask `k-1` in \(x_i\). The dot
in `.'` matters: it is a transpose without complex conjugation.

Each packed eigenvector is valid, but its phase and its basis inside a
repeated eigenspace are arbitrary. Do not compare two raw eigenvector columns
entry by entry.

## Main outputs

| Field | Meaning |
| --- | --- |
| `S.blockLeftRegularMatrix` | The \(rN\times rN\) complex matrix \(\Lambda_r(A)\). |
| `S.eigenvalues` | All raw complex eigenvalues, including full-regular multiplicities. |
| `S.packedRightEigenvectors` | Complex columns satisfying \(\Lambda_r(A)v_j=\zeta_jv_j\). |
| `S.rightEigenvectorCoefficients` | Unpacked `r-by-N-by-(r*N)` Clifford coefficient data. |
| `S.relativeEigenResiduals` | Scaled residual for every raw eigenpair. |
| `S.sigmaMin`, `S.sigmaMax` | Smallest and largest singular values of the packed action. |
| `S.conditionNumber2` | Thresholded \(2\)-norm condition number; `Inf` when the numerical rank test says singular. |
| `S.isNumericallyStarSelfAdjoint` | Whether \(A\approx A^\star\). |
| `S.hermitian` | Sorted real eigensystem and positive-definite diagnostics for the symmetrized matrix, when available. |

`S.leftRegularMatrix` is retained as an alias of
`S.blockLeftRegularMatrix` for continuity with the one-multivector utility.

## Hermitian matrices, Gram matrices, and \(\lambda_{\min}\)

For a Clifford matrix,

\[
(A^\star)_{ij}=(a_{ji})^\star.
\]

The block map respects this adjoint:

\[
\Lambda_r(A^\star)=\Lambda_r(A)^H.
\]

Therefore, when \(A=A^\star\), the full block map is Hermitian. In that
case the nested fields

    S.hermitian.lambdaMin
    S.hermitian.lambdaMax
    S.hermitian.positiveDefiniteConditionNumber2

are the right quantities for the full coefficient-packed action. In
particular,

\[
\lambda_{\min}(\Lambda_r(A))>0
\]

is exactly the finite-dimensional positive-definiteness test, and

\[
\kappa_2(\Lambda_r(A))=
\frac{\lambda_{\max}}{\lambda_{\min}}
\]

on the positive-definite Hermitian branch.

For a nonnormal matrix there is no meaningful generic “smallest eigenvalue.”
Use `S.sigmaMin` and `S.conditionNumber2`, not `min(abs(S.eigenvalues))`, to
measure invertibility and linear-solve stability.

If numerical self-adjointness is detected only up to tolerance, the Hermitian
spectral and positive-definite fields in `S.hermitian` describe

\[
h=\frac{A+A^\star}{2},
\]

while the fields directly under `S` always describe your original input
\(A\). The nested fields explicitly prefixed `originalMatrix` also report
residuals against that original input.

For transparency, `S.hermitian.symmetrizedBlockLeftRegularMatrix` is the
actual represented map \(\Lambda_r(h)\). The eigensolver uses
`S.hermitian.numericallyHermitianMatrix`, its explicit Hermitian projection
\((\Lambda_r(h)+\Lambda_r(h)^H)/2\), solely to remove roundoff-level
non-Hermiticity before sorting real eigenvalues. The two matrices agree to the
reported `symmetrizedMapHermitianResidual` scale.

## Repeated values are expected

Put \(d=2^{\lfloor m/2\rfloor}\). The metadata in
`S.regularRepresentation` records the forced copies in this block
left-regular representation.

- For even \(m\), \(\mathbb C\otimes\mathrm{Cl}_{0,m}\cong M_d(\mathbb C)\),
  and the full block map contains \(d\) copies of an \(rd\)-dimensional
  irreducible action.
- For odd \(m\), the algebra has two simple sectors. A faithful minimal
  representation contains both \(rd\)-dimensional sectors, and the full
  block map contains \(d\) copies of each sector.

So repeated raw values are normal even before geometry creates any extra
degeneracy. `S.eigenvalueClusters` reports numerical groups but never removes,
deduplicates, or divides multiplicities.

The full block map is faithful but is not onto arbitrary complex
`(r*N)-by-(r*N)` matrices: every `N-by-N` block must retain left-regular
Clifford structure. It is also not the later compact \(\tau_m\) map.

## Tolerances versus a design epsilon

The default relative numerical resolution is

\[
\tau_{\rm rel}=100(rN)\varepsilon_{\rm mach},
\]

and the default scale-dependent threshold is

\[
\tau=\tau_{\rm abs}+\tau_{\rm rel}\sigma_{\max}.
\]

The optional fields are:

| Option | Role |
| --- | --- |
| `relativeTolerance`, `absoluteTolerance` | Base numerical resolution. |
| `singularValueTolerance` | Explicit rank cutoff. |
| `eigenvalueClusterTolerance` | Maximum numerical-cluster diameter; it never changes raw values. |
| `selfAdjointTolerance`, `normalityTolerance` | Dimensionless structural tests. |
| `positiveDefiniteTolerance` | Safety cutoff for declaring \(\lambda_{\min}>0\). |

This numerical tolerance is **not** the same as a scientific or optimization
constraint such as \(\lambda_{\min}\ge\epsilon_{\rm design}\). Choose the
latter from the amplification or condition number you can tolerate. For
example, after computing `S.hermitian.lambdaMax`, a desired Hermitian condition
limit \(\kappa_{\rm target}\) suggests

\[
\epsilon_{\rm design}\ge
\frac{\lambda_{\max}}{\kappa_{\rm target}}.
\]

## Scope of this milestone

This update returns eigenvectors as Clifford coefficient tuples and provides
the Hermitian Gram-matrix diagnostics needed now. It intentionally does not
yet return spectral projectors or inverse square roots as Clifford coefficient
tensors: a public, independently tested block-image decoder/validator is not
yet exposed, and arbitrary numerical eigenspace projectors need not lie in the
block-map image. That is a sensible next algebra utility after this eigensystem
milestone.

## Merge and test

This archive has no enclosing `clifford-adjoint-matlab` folder. Copy or unzip
its contents into the root of your existing repository. When Finder asks about
`src` and `tests`, choose **Merge**, never Replace.

Open `run_matrix_eigensystem_test.m` in MATLAB and press Run. Expected output:

    Cl_{0,4}: 32-by-32 Clifford-matrix eigensystem; packed spectral diagnostics passed.
    Cl_{0,5}: 64-by-64 Clifford-matrix eigensystem; packed spectral diagnostics passed.

The test independently builds the packed action from direct Clifford products,
checks every raw and unpacked eigenvector, validates cell and tensor inputs,
tests \(m=0\) and \(m=1\), confirms the nonnormal-unipotent warning, verifies
singularity/rank thresholds, and checks the self-star positive-definite Gram
case and its \(\lambda_{\min}\) diagnostics.
