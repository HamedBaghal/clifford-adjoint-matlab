# Eighth local test: block left-regular decoder and image validator

This update adds the reverse direction of the existing block left-regular
map.  It takes an ordinary complex matrix, determines whether it genuinely
has the required Clifford block structure, and recovers the coefficient
tensor when it does.

It requires the earlier modules already in the repository:

- `clifford_algebra.m`
- `clifford_left_regular_matrix.m`
- `clifford_matrix_left_regular_matrix.m`
- `clifford_multivector_from_left_regular_matrix.m` (used by the test for a
  one-entry consistency check)

## New files

- `src/clifford_matrix_from_left_regular_matrix.m`
- `tests/test_clifford_matrix_from_left_regular_matrix.m`
- `run_block_decoder_test.m`

## The problem it solves

For \(N=2^m\), the existing encoder maps a Clifford-valued matrix

\[
A=[a_{ij}]\in M_{r,s}(\mathbb C\otimes\operatorname{Cl}_{0,m})
\]

to the ordinary complex block matrix

\[
\Lambda_{r,s}(A)=[\lambda_m(a_{ij})]\in M_{rN,sN}(\mathbb C).
\]

This map is faithful, but it is not onto.  An arbitrary \(N\)-by-\(N\)
complex block does not necessarily equal \(\lambda_m(a)\) for a
multivector \(a\).  In fact, one block has only \(N\) complex Clifford
coefficients but contains \(N^2\) complex matrix entries.

The new function is

```matlab
[entries, info] = clifford_matrix_from_left_regular_matrix(B, alg)
```

It accepts a nonempty finite complex matrix `B` whose dimensions are
multiples of `alg.nBlades`.  If `B` has size `(r*N)`-by-`(s*N)`, then
`entries` has canonical size `r`-by-`s`-by-`N`, with

\[
\texttt{entries(i,j,k)}=
\text{coefficient of blade mask }k-1\text{ in }a_{ij}.
\]

For \(m=0\), \(N=1\), so every ordinary complex matrix is already in the
image and MATLAB displays the singleton last dimension as an ordinary
two-dimensional matrix.

## Why this is stronger than reading first columns

For an exact left-regular block, its first column is indeed the coefficient
vector:

\[
\lambda_m(a)(:,1)=\operatorname{coeff}(a).
\]

So first-column extraction is an exact inverse *on the image*.  It is not a
safe way to process a general numerical complex matrix: a non-Clifford
perturbation can leave every first column unchanged while corrupting other
columns.  The test includes exactly this trap.

Instead, the decoder uses all entries of every block.  Put

\[
E_p=\lambda_m(e_{p-1}),\qquad p=1,\ldots,N.
\]

The signed-permutation matrices \(E_p\) are Frobenius orthogonal:

\[
\langle E_p,E_q\rangle_F=N\,\delta_{pq}.
\]

For a candidate block \(B_{ij}\), the recovered coefficient is therefore

\[
\widehat a_{ij,p}
=\frac{1}{N}\operatorname{tr}(E_p^H B_{ij}).
\]

The reconstructed matrix `info.reconstructedBlockLeftRegularMatrix` is the
unique Frobenius-nearest block left-regular matrix to the input.  Thus, for
the full block map,

\[
\|\Lambda_{r,s}(A)\|_F^2
=N\sum_{i,j,p}|a_{ij,p}|^2.
\]

This is both a decoder and an orthogonal structure projector.  It is useful
when a later ordinary-complex calculation (for example, a matrix function)
is expected to preserve Clifford structure but must be checked numerically.

## Strict validation and diagnostic mode

The default mode is strict:

```matlab
[entries, info] = clifford_matrix_from_left_regular_matrix(B, alg);
```

It succeeds only when the full reconstruction residual satisfies

\[
\texttt{absoluteResidual}
\leq \texttt{absoluteTolerance}
+ \texttt{relativeTolerance}\,
   \max(1,\|B\|_F).
\]

The default relative tolerance is

```matlab
100 * eps * max(1, max(size(B)))
```

and the default absolute tolerance is zero.  This is a conservative
floating-point check for results that should already be structured; it is
not a mathematical claim that a noisy input has become exact.  For a
single-precision calculation or deliberately approximate data, choose a
tolerance reflecting that input error.  These decoder tolerances are also
different from the design parameter \(\epsilon\) used to prescribe a Gram
matrix lower-eigenvalue margin: here they only decide whether a computed
complex matrix is close enough to the block left-regular image to decode.

To inspect an arbitrary complex matrix without an error, choose report mode:

```matlab
options = struct('validationMode', 'report');
[approxEntries, info] = ...
    clifford_matrix_from_left_regular_matrix(B, alg, options);
```

Then `approxEntries` describes the nearest structured approximation, and
`info.isInBlockLeftRegularImage` says whether the original `B` passed the
tolerance.  Important diagnostic fields are:

- `absoluteResidual`, `relativeResidual`, and `acceptanceThreshold`;
- `blockAbsoluteResiduals` and `blockRelativeResiduals`, which locate bad
  Clifford blocks;
- `reconstructedBlockLeftRegularMatrix`, the nearest structured block map;
- `frobeniusScaling = sqrt(N)`, encoding the norm identity above.

You can set a custom tolerance either with a structure or, in strict mode,
as a scalar shorthand for `relativeTolerance`:

```matlab
% Accept only a relative reconstruction residual at most 1e-10.
[entries, info] = clifford_matrix_from_left_regular_matrix(B, alg, 1e-10);

% Combine an absolute and relative tolerance while keeping report mode.
options = struct('validationMode', 'report', ...
                 'absoluteTolerance', 1e-12, ...
                 'relativeTolerance', 1e-10);
[approxEntries, info] = ...
    clifford_matrix_from_left_regular_matrix(B, alg, options);
```

If report mode returns `false` for `info.isInBlockLeftRegularImage`, do not
interpret `approxEntries` as an exact decoded Clifford matrix.  It is the
best approximation in Frobenius norm, which is often valuable for
diagnosis, but the residual must be appropriate for the numerical task.

## Relation to the eigensystem utility

The existing matrix eigensystem utility works with the full ordinary complex
reference matrix.  This decoder gives us a principled way to bring an object
back only when it has the required block structure.  In particular, a
complete spectral-cluster projector or a positive inverse square root that
is obtained by a genuine Clifford matrix functional calculus should pass
this test (up to rounding).

An individual rank-one projector `v*v'` formed from one ordinary complex
eigenvector is generally **not** guaranteed to be a block left-regular
matrix: the regular representation can have multiplicities.  That is why a
future functional-calculus utility must validate the full result before
decoding it, rather than decoding arbitrary eigenvector-derived matrices.

## Merge and test

This archive has no enclosing `clifford-adjoint-matlab` folder.  Extract its
contents into the root of your existing repository.  When Finder asks about
`src` and `tests`, choose **Merge**, never Replace.

Open `run_block_decoder_test.m` in MATLAB and press **Run**.  Expected output
is:

```text
Cl_{0,4}: 32-by-48 complex block map; decoder and image validation passed.
Cl_{0,5}: 64-by-96 complex block map; decoder and image validation passed.
```

The test covers exact rectangular recovery, agreement with the established
one-multivector decoder, an independent Hilbert--Schmidt projection oracle,
a corruption invisible to first-column extraction, strict/report modes,
absolute/relative/mixed tolerances, and the edge dimensions \(m=0\) and
\(m=1\).
