function leftRegular = clifford_left_regular_matrix(a, alg)
%CLIFFORD_LEFT_REGULAR_MATRIX Faithful complex left-regular representation.
%
%   LEFTREGULAR = CLIFFORD_LEFT_REGULAR_MATRIX(A, ALG) returns the dense
%   complex matrix of left multiplication by the multivector A in the
%   coefficient basis of the complexification of Cl_{0,m}.  If N=2^m, then
%   LEFTREGULAR is N-by-N and its MATLAB column J is the coefficient vector
%   of
%
%       A * e_(J-1),
%
%   where J-1 is the zero-based blade mask.  In particular,
%
%       LEFTREGULAR(:, 1) = A.
%
%   With lambda(A) denoting this matrix, the map is a faithful unital
%   complex *-representation:
%
%       lambda(AB) = lambda(A)*lambda(B),
%       lambda(A^star) = lambda(A)',
%       lambda(1) = eye(N).
%
%   Here AB denotes the Clifford product, A^star is
%   CLIFFORD_MULTIVECTOR_STAR(A, ALG), and the apostrophe in the second
%   identity is the ordinary complex conjugate transpose relative to the
%   orthonormal blade-coefficient basis.  The representation is injective,
%   but it is not onto all N-by-N complex
%   matrices.  Use CLIFFORD_MULTIVECTOR_FROM_LEFT_REGULAR_MATRIX only to
%   decode a matrix that is known (up to numerical error) to be in this
%   left-regular image.
%
%   This is the canonical reference representation.  It has 4^m matrix
%   entries, so it is practical only for moderate dimensions even though the
%   blade metadata itself supports larger m.

    local_validate_algebra(alg);
    nBlades = alg.nBlades;
    a = local_validate_multivector(a, nBlades);
    a = double(a);

    leftRegular = zeros(nBlades, nBlades);
    nonzeroA = find(a ~= 0);

    for rightIndex = 1:nBlades
        for position = 1:numel(nonzeroA)
            leftIndex = nonzeroA(position);
            [signValue, outputIndex] = ...
                alg.multiplyIndices(leftIndex, rightIndex);

            % For a fixed right blade, leftIndex -> outputIndex is a
            % bijection.  Assignment therefore gives one coefficient per row.
            leftRegular(outputIndex, rightIndex) = ...
                signValue * a(leftIndex);
        end
    end
end

function local_validate_algebra(alg)
    isValid = isstruct(alg) && isscalar(alg) && isfield(alg, 'm') && ...
        isfield(alg, 'nBlades') && isfield(alg, 'multiplyIndices') && ...
        isa(alg.multiplyIndices, 'function_handle') && ...
        isnumeric(alg.m) && isreal(alg.m) && isscalar(alg.m) && ...
        isfinite(alg.m) && alg.m == floor(alg.m) && ...
        alg.m >= 0 && alg.m <= 52 && ...
        isnumeric(alg.nBlades) && isreal(alg.nBlades) && ...
        isscalar(alg.nBlades) && isfinite(alg.nBlades) && ...
        alg.nBlades == 2^alg.m;

    if ~isValid
        error('clifford_left_regular_matrix:InvalidAlgebra', ...
            ['ALG must be a structure returned by clifford_algebra, with ', ...
             'a valid multiplyIndices function handle.']);
    end
end

function coefficientVector = local_validate_multivector(value, nBlades)
    if ~(isnumeric(value) && isvector(value) && numel(value) == nBlades)
        error('clifford_left_regular_matrix:InvalidCoefficientVector', ...
            'A must be a numeric vector with exactly ALG.nBlades entries.');
    end

    coefficientVector = value(:);
    if ~all(isfinite(coefficientVector))
        error('clifford_left_regular_matrix:NonFiniteCoefficient', ...
            'A contains a NaN or Inf coefficient.');
    end
end
