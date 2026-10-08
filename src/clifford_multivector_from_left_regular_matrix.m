function [a, relativeResidual] = clifford_multivector_from_left_regular_matrix(leftRegular, alg, tolerance)
%CLIFFORD_MULTIVECTOR_FROM_LEFT_REGULAR_MATRIX Decode a left-regular matrix.
%
%   [A, RELATIVERESIDUAL] =
%   CLIFFORD_MULTIVECTOR_FROM_LEFT_REGULAR_MATRIX(LEFTREGULAR, ALG) recovers
%   the multivector A from a matrix in the image of
%   CLIFFORD_LEFT_REGULAR_MATRIX.  The function checks that LEFTREGULAR is
%   actually left-regular; it does not silently treat an arbitrary complex
%   matrix as a Clifford multivector.
%
%   The recovery is A = LEFTREGULAR(:,1), because left multiplication by A
%   maps the scalar blade 1 to A.  The returned relative Frobenius residual
%   verifies all columns.  By default, a matrix is accepted when
%
%       RELATIVERESIDUAL <= 100*eps*max(1, ALG.nBlades).
%
%   Supply a nonnegative scalar TOLERANCE as the third input to choose a
%   different acceptance threshold.  This function is the inverse only on
%   the left-regular image, not on all N-by-N complex matrices.

    local_validate_algebra(alg);
    nBlades = alg.nBlades;
    if nargin < 3
        tolerance = 100 * eps * max(1, nBlades);
    end

    local_validate_matrix(leftRegular, nBlades);
    local_validate_tolerance(tolerance);

    a = full(double(leftRegular(:, 1)));
    nonzeroA = find(a ~= 0);
    squaredResidual = 0;

    % Verify the image condition a column at a time, avoiding an extra
    % N-by-N reconstruction when LEFTREGULAR is already a large matrix.
    for rightIndex = 1:nBlades
        expectedColumn = zeros(nBlades, 1);
        for position = 1:numel(nonzeroA)
            leftIndex = nonzeroA(position);
            [signValue, outputIndex] = ...
                alg.multiplyIndices(leftIndex, rightIndex);
            expectedColumn(outputIndex) = signValue * a(leftIndex);
        end
        difference = double(leftRegular(:, rightIndex)) - expectedColumn;
        squaredResidual = squaredResidual + sum(abs(difference).^2);
    end

    relativeResidual = sqrt(squaredResidual) / ...
        max(1, norm(leftRegular, 'fro'));
    if relativeResidual > tolerance
        error('clifford_multivector_from_left_regular_matrix:NotInLeftRegularImage', ...
            ['LEFTREGULAR is not within the requested tolerance of the ', ...
             'left-regular Clifford image.']);
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
        error('clifford_multivector_from_left_regular_matrix:InvalidAlgebra', ...
            ['ALG must be a structure returned by clifford_algebra, with ', ...
             'a valid multiplyIndices function handle.']);
    end
end

function local_validate_matrix(leftRegular, nBlades)
    hasCorrectShape = isnumeric(leftRegular) && ismatrix(leftRegular) && ...
        size(leftRegular, 1) == nBlades && ...
        size(leftRegular, 2) == nBlades;
    if ~hasCorrectShape
        error('clifford_multivector_from_left_regular_matrix:InvalidMatrix', ...
            'LEFTREGULAR must be a numeric ALG.nBlades-by-ALG.nBlades matrix.');
    end
    if ~all(isfinite(leftRegular(:)))
        error('clifford_multivector_from_left_regular_matrix:NonFiniteEntry', ...
            'LEFTREGULAR contains a NaN or Inf entry.');
    end
end

function local_validate_tolerance(tolerance)
    if ~(isnumeric(tolerance) && isreal(tolerance) && isscalar(tolerance) && ...
            isfinite(tolerance) && tolerance >= 0)
        error('clifford_multivector_from_left_regular_matrix:InvalidTolerance', ...
            'TOLERANCE must be one finite nonnegative real scalar.');
    end
end
