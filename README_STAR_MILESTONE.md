# Third local test: the complex Clifford adjoint

This update adds the multivector operation needed before constructing a
complex-adjoint representation.  It requires the two earlier modules:
`clifford_algebra.m` and `clifford_multivector_product.m`.

## New files

- `src/clifford_multivector_star.m`
- `tests/test_clifford_multivector_star.m`
- `run_star_test.m`

## Definition

For a multivector

\[
A=\sum_I a_I e_I \in \mathbb{C}\otimes\mathrm{Cl}_{0,m},
\]

the new function computes

\[
A^\star
=\sum_I \overline{a_I}
(-1)^{|I|(|I|+1)/2}e_I.
\]

It is the complex extension of Clifford conjugation.  In particular,

\[
e_i^\star=-e_i,\qquad
(AB)^\star=B^\star A^\star,\qquad
(A^\star)^\star=A.
\]

This is **not** merely reversion: coefficients are complex-conjugated and the
order of a product is reversed.  This is the involution required by a future
Clifford-valued matrix adjoint, where the matrix adjoint will be entrywise
star followed by matrix transpose.

## Merge and test

This archive has no enclosing `clifford-adjoint-matlab` folder.  Copy or unzip
its contents **into the root of your existing repository**.  When Finder asks
about the existing `src` and `tests` folders, choose **Merge**, never Replace.

Then open `run_star_test.m` in MATLAB and press **Run**.  Expected output:

```text
Cl_{0,4}: 16 coefficient components; complex Clifford star passed.
Cl_{0,5}: 32 coefficient components; complex Clifford star passed.
```

The test uses a grade-sign reference independent of the implementation.  It
also checks conjugate-linearity, involution, the reversed-product law, the
matrix identity \(L_{A^\star}=L_A^H\), positivity of the scalar part of
\(A^\star A\), and invalid inputs.
