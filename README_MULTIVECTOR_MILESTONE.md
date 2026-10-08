# Second local test: complex Clifford multivector products

This update adds the geometric product of two full coefficient vectors in
\(\mathrm{Cl}_{0,m}\).  It builds on the previously tested
`clifford_algebra.m` file.

## New files

- `src/clifford_multivector_product.m`
- `tests/test_clifford_multivector_product.m`
- `run_multivector_test.m`

## Input convention

If `a` is a coefficient vector, `a(mask+1)` is the coefficient of the blade
represented by `mask`.  For example, in every dimension `a(1)` is the scalar
coefficient and `a(2)` is the \(e_1\) coefficient.  Inputs may be real or
complex; the product is complex-bilinear.

The product uses only nonzero coefficient pairs and does not allocate a
\(4^m\) multiplication table.  A dense multivector nevertheless has
\(2^m\) entries, so the practical dimension remains moderate.

## Merge and test

This archive has no enclosing `clifford-adjoint-matlab` folder.  Copy or unzip
its contents **into the root of your existing repository**, so the new source
file lands in its existing `src` folder and the test lands in its existing
`tests` folder.

Then open `run_multivector_test.m` in MATLAB and press **Run**.  Expected
output:

```text
Cl_{0,4}: 16 coefficient components; complex multivector product passed.
Cl_{0,5}: 32 coefficient components; complex multivector product passed.
```

The test checks every pair of basis blades against an independent matrix
realization built directly from the generator rules, then checks complex
bilinearity, the left-regular representation, associativity, the scalar unit,
the zero multivector, and invalid inputs.  This is intentionally stronger than
checking the product routine against the same blade-multiplication helper it
uses internally.
