# First local test: arbitrary-dimensional Clifford blades

This small first milestone implements the real Clifford algebra
\(\mathrm{Cl}_{0,m}\) for arbitrary `m` with

\[
e_i^2=-1,\qquad e_i e_j=-e_j e_i\quad(i\ne j).
\]

It contains no dimension-specific `m=4` code and no symbolic toolbox code.

## Files

- `src/clifford_algebra.m` — generic blade encoding, blade products, grades,
  reversion, and Clifford conjugation.
- `tests/test_clifford_algebra.m` — exact tests for `m=4` and `m=5`.
- `run_first_test.m` — the easiest entry point.

## Run it in MATLAB

Copy the three files into the identically named locations in the local
`clifford-adjoint-matlab` folder.  Then either open `run_first_test.m` and
press **Run**, or enter this in the MATLAB Command Window from the repository
root:

```matlab
addpath('src', 'tests');
results = test_clifford_algebra([4, 5])
```

Expected output:

```text
Cl_{0,4}: 16 blades, 4096 associativity cases passed.
Cl_{0,5}: 32 blades, 32768 associativity cases passed.
```

For an individual dimension, for example `m=5`, use:

```matlab
alg = clifford_algebra(5);
e1 = uint64(1);
e2 = uint64(2);
[signValue, productMask] = alg.multiplyMasks(e1, e2)
```

The answer is `signValue = 1` and `productMask = 3`, representing
\(e_1 e_2=e_1e_2\).

## Why the encoding is general

Each blade is represented by a `uint64` bit mask.  The least-significant bit
represents \(e_1\), the next bit represents \(e_2\), and so on.  Thus a
blade mask is independent of dimension-specific names such as `e12` or
`e1234`.

The blade-product rule itself works for all supported `m`.  The later stages
that store full coefficient vectors and matrix representations necessarily
grow exponentially with `m`; the present test is deliberately small and fast
for `m=4` and `m=5`.

`alg.starBlade` acts on a basis blade.  When multivector coefficients are
introduced later, their star operation must additionally complex-conjugate
the coefficients.
