# Sixth local test: block left-regular matrix map

This update turns a matrix whose entries are complex Clifford multivectors
into one ordinary complex block matrix. It requires the earlier modules:

- clifford_algebra.m
- clifford_multivector_product.m
- clifford_multivector_star.m
- clifford_left_regular_matrix.m

## New files

- src/clifford_matrix_left_regular_matrix.m
- tests/test_clifford_matrix_left_regular_matrix.m
- run_block_matrix_test.m

## The map

Let \(N=2^m\), and let

\[
A=[a_{ij}]\in M_{r,s}(\mathbb C\otimes\mathrm{Cl}_{0,m}).
\]

The function constructs

\[
\Lambda_{r,s}(A)=
\begin{bmatrix}
\lambda_m(a_{11}) & \cdots & \lambda_m(a_{1s})\\
\vdots & \ddots & \vdots\\
\lambda_m(a_{r1}) & \cdots & \lambda_m(a_{rs})
\end{bmatrix}
\in M_{rN,sN}(\mathbb C),
\]

where \(\lambda_m\) is the already-tested left-regular map for one
multivector. The new MATLAB function is

    [L, info] = clifford_matrix_left_regular_matrix(entries, alg);

The output L is the complex matrix that later eigenvalue, singular-value, and
linear-system routines will use.

## How to enter a Clifford-valued matrix

### Recommended for first use: a cell matrix

Each cell is one coefficient vector of length N.

    alg = clifford_algebra(4);
    N = alg.nBlades;

    one = zeros(N, 1);  one(1) = 1;
    e1  = zeros(N, 1);  e1(2)  = 1;
    e2  = zeros(N, 1);  e2(3)  = 1;

    A = cell(2, 2);
    A{1,1} = 2 * one + 1i * e1;
    A{1,2} = e2;
    A{2,1} = -e1;
    A{2,2} = one;

    L = clifford_matrix_left_regular_matrix(A, alg);

Here A is a \(2\times2\) Clifford-valued matrix, while L is an ordinary
\(32\times32\) complex matrix because \(N=16\) for \(m=4\).

### Efficient numeric form: an r-by-s-by-N tensor

The canonical numeric form has the Clifford-matrix row and column first:

\[
\texttt{entries(i,j,k)}
=\text{coefficient of the blade with mask }k-1\text{ in }a_{ij}.
\]

For example, to put the same coefficient vector aij in position \((i,j)\):

    entries = zeros(r, s, N);
    entries(i, j, :) = reshape(aij, [1, 1, N]);

The function intentionally uses r-by-s-by-N, not N-by-r-by-s. This keeps the
first two indices equal to the ordinary matrix indices. For \(m=0\), where
\(N=1\), an ordinary complex r-by-s matrix is accepted directly.

The second output, info, records rowCount, columnCount, nBlades, outputSize,
inputFormat, and the exact packing convention.

## What the block matrix does

If \(X=(x_1,\ldots,x_s)^T\) is a column of Clifford multivectors, pack its
coefficient vectors as

\[
\operatorname{pack}(X)=
\begin{bmatrix}
\operatorname{coeff}(x_1)\\
\vdots\\
\operatorname{coeff}(x_s)
\end{bmatrix}.
\]

Then

\[
\operatorname{pack}(AX)=\Lambda_{r,s}(A)\operatorname{pack}(X).
\]

So block rows correspond to the output Clifford-matrix row, and each block
uses the same blade order as the earlier single-multivector functions.

For compatible Clifford-valued matrices,

\[
\Lambda(AB)=\Lambda(A)\Lambda(B).
\]

For a square Clifford matrix, with

\[
(A^\star)_{ij}=(a_{ji})^\star,
\]

the ordinary complex adjoint satisfies

\[
\Lambda(A^\star)=\Lambda(A)^H.
\]

The transpose of block positions is essential here: applying star to each
entry without transposing the matrix is not the matrix adjoint.

The map is faithful. Every entry \(a_{ij}\) is the first column of its
\(N\times N\) block, so \(\Lambda(A)=0\) implies \(A=0\). It is not onto all
complex matrices: every block must separately have left-regular Clifford
structure.

For square A, A is two-sided invertible exactly when \(\Lambda(A)\) is
nonsingular. Therefore \(\sigma_{\min}(\Lambda(A))\) and
\(\kappa_2(\Lambda(A))\) are exact stability diagnostics for this packed
full left-regular action, measured in the Euclidean norm of coefficient
vectors. They are not yet diagnostics for a compact \(\tau_m\) realization.

## Relation to the eigensystem utility

The previous eigensystem utility takes one multivector, not a tensor or cell
matrix. This new map is the correct bridge to the matrix problem:

    L = clifford_matrix_left_regular_matrix(A, alg);
    [V, D] = eig(L);

Each column of V then represents a packed tuple of Clifford multivectors.
These are eigenpairs of the full packed left-module problem; their
multiplicities can reflect the full regular representation and should not yet
be called a unique intrinsic Clifford spectrum. The next package module will
wrap this carefully as a dedicated Clifford-matrix eigensystem utility,
including the Hermitian/stability diagnostics already developed for one
multivector.

This is the full, faithful reference representation. It is not the later
compact \(\tau_m\) representation. Its complex size grows from r-by-s to
\((r2^m)\)-by-\((s2^m)\), so use moderate m and matrix sizes.

## Merge and test

This archive has no enclosing clifford-adjoint-matlab folder. Copy or unzip
its contents into the root of your existing repository. When Finder asks
about src and tests, choose Merge, never Replace.

Open run_block_matrix_test.m in MATLAB and press Run. Expected output:

    Cl_{0,4}: 2-by-3 Clifford matrix -> 32-by-48 complex block map passed.
    Cl_{0,5}: 2-by-3 Clifford matrix -> 64-by-96 complex block map passed.

The test uses an independent signed-permutation block oracle and verifies
tensor and cell inputs, packed block action, multiplication homomorphism,
complex linearity, block star-adjoint structure, scalar Kronecker structure,
identity, trace/Frobenius invariants, edge dimensions, and invalid inputs.
