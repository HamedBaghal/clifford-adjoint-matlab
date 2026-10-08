# Fourth local test: faithful complex left-regular representation

This update turns one complex Clifford multivector into a standard complex
matrix.  It requires the three earlier modules:

- `clifford_algebra.m`
- `clifford_multivector_product.m`
- `clifford_multivector_star.m`

## New files

- `src/clifford_left_regular_matrix.m`
- `src/clifford_multivector_from_left_regular_matrix.m`
- `tests/test_clifford_left_regular_matrix.m`
- `run_left_regular_test.m`

## Representation

For $A\in\mathbb{C}\otimes\mathrm{Cl}_{0,m}$, the function constructs

\[
\lambda_m(A):X\longmapsto AX,
\]

in the ordered blade-coefficient basis.  If $N=2^m$, the result is an
$N\times N$ complex matrix with column $j$ equal to the coefficient vector
of $A e_{j-1}$.  In particular, its first column is exactly the coefficient
vector of $A$.

It satisfies

\[
\lambda_m(AB)=\lambda_m(A)\lambda_m(B),\qquad
\lambda_m(A^\star)=\lambda_m(A)^H.
\]

Therefore this is a faithful unital \(*\)-representation.  It is an
isomorphism **onto its image**, not onto every (N\times N) complex matrix.
The decoder `clifford_multivector_from_left_regular_matrix` deliberately
rejects matrices outside that image.

## Why this comes before the compact complex adjoint

This is the safe canonical reference map.  It is not the smallest possible
complex representation: it is $16\times16$ for $m=4$ and $32\times32$
for $m=5$.  A future reduced map $\tau_m$ will be smaller, but must handle
the odd-dimensional direct-sum structure correctly.  We will test that later
against this faithful representation.

The matrix stores $4^m$ entries, so use it only for moderate $m$.  Its
ordinary complex eigenvalues are useful for numerical work, but can include
representation-theoretic multiplicities.  For a nonnormal matrix, stability
should be assessed with singular values rather than only eigenvalues.

## Merge and test

This archive has no enclosing `clifford-adjoint-matlab` folder.  Copy or unzip
its contents **into the root of your existing repository**.  When Finder asks
about `src` and `tests`, choose **Merge**, never Replace.

Open `run_left_regular_test.m` in MATLAB and press **Run**.  Expected output:

```text
Cl_{0,4}: 16-by-16 left-regular matrix; faithful representation passed.
Cl_{0,5}: 32-by-32 left-regular matrix; faithful representation passed.
```

The test compares every blade matrix with an independently constructed
signed-permutation oracle.  It also verifies the action, product homomorphism,
star-to-Hermitian-adjoint identity, matrix decoder, generator relations, and
invalid inputs.
