function c = clifford_multivector_product(a, b, alg)
%CLIFFORD_MULTIVECTOR_PRODUCT Geometric product of two Cl_{0,m} multivectors.
%
%   C = CLIFFORD_MULTIVECTOR_PRODUCT(A, B, ALG) returns the geometric product
%   C = A*B in the complexification of the Clifford algebra Cl_{0,m}.  ALG
%   must be the structure returned by CLIFFORD_ALGEBRA(M).
%
%   A and B are numeric vectors of length ALG.nBlades.  Their coefficient at
%   MATLAB index MASK+1 multiplies the canonical blade encoded by MASK; hence
%
%       A(1)  is the scalar coefficient,
%       A(2)  is the e_1 coefficient,
%       A(4)  is the e_1*e_2 coefficient,
%       ...
%
%   C is returned as a double column vector.  The implementation evaluates only the
%   nonzero coefficient pairs and does not build a 4^m multiplication table.
%   It is therefore the correct general implementation for moderate m.  Dense
%   multivectors nevertheless contain 2^m coefficients, so very high
%   dimensions require a more specialised sparse or matrix representation.

    local_validate_algebra(alg);
    nBlades = alg.nBlades;
    a = local_validate_multivector(a, nBlades, 'a', ...
        'InvalidLeftCoefficientVector');
    b = local_validate_multivector(b, nBlades, 'b', ...
        'InvalidRightCoefficientVector');

    % Convert integer/single inputs to double before complex accumulation.
    a = double(a);
    b = double(b);
    c = zeros(nBlades, 1);
    nonzeroA = find(a ~= 0);
    nonzeroB = find(b ~= 0);

    for leftPosition = 1:numel(nonzeroA)
        leftIndex = nonzeroA(leftPosition);
        leftCoefficient = a(leftIndex);

        for rightPosition = 1:numel(nonzeroB)
            rightIndex = nonzeroB(rightPosition);
            [signValue, productIndex] = ...
                alg.multiplyIndices(leftIndex, rightIndex);
            c(productIndex) = c(productIndex) + ...
                signValue * leftCoefficient * b(rightIndex);
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
        error('clifford_multivector_product:InvalidAlgebra', ...
            ['ALG must be a structure returned by clifford_algebra, with ', ...
             'a valid multiplyIndices function handle.']);
    end
end

function coefficientVector = local_validate_multivector(value, nBlades, name, errorId)
    if ~(isnumeric(value) && isvector(value) && numel(value) == nBlades)
        error(['clifford_multivector_product:', errorId], ...
            '%s must be a numeric vector with exactly ALG.nBlades entries.', name);
    end

    coefficientVector = value(:);
    if ~all(isfinite(coefficientVector))
        error('clifford_multivector_product:NonFiniteCoefficient', ...
            '%s contains a NaN or Inf coefficient.', name);
    end
end
