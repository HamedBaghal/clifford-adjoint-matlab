function aStar = clifford_multivector_star(a, alg)
%CLIFFORD_MULTIVECTOR_STAR Complex Clifford adjoint of a multivector.
%
%   ASTAR = CLIFFORD_MULTIVECTOR_STAR(A, ALG) returns the complex Clifford
%   adjoint A^star in the complexification of Cl_{0,m}.  ALG must be the
%   structure returned by CLIFFORD_ALGEBRA(M), using
%
%       e_i^2 = -1,              e_i e_j = -e_j e_i  (i ~= j).
%
%   If A = sum_I a_I e_I and r = |I| is the grade of e_I, then
%
%       A^star = sum_I conj(a_I) * (-1)^(r*(r + 1)/2) * e_I.
%
%   Thus STAR is conjugate-linear and anti-multiplicative:
%
%       (alpha*A + beta*B)^star = conj(alpha)*A^star + conj(beta)*B^star,
%       (A*B)^star = B^star*A^star.
%
%   In particular, e_i^star = -e_i.  This is the complex extension of
%   Clifford conjugation, not merely reversion.  The coefficient at MATLAB
%   index MASK+1 corresponds to the canonical blade encoded by MASK.  A may
%   be a real or complex row or column vector of length ALG.nBlades; ASTAR is
%   always returned as a double column vector.

    local_validate_algebra(alg);
    nBlades = alg.nBlades;
    a = local_validate_multivector(a, nBlades);
    a = double(a);

    aStar = zeros(nBlades, 1);
    for index = 1:nBlades
        % Existing blade-level star is Clifford conjugation.  It supplies
        % the grade sign and would also support a future blade permutation.
        [signValue, outputMask] = alg.starBlade(uint64(index - 1));
        outputIndex = double(outputMask) + 1;
        aStar(outputIndex) = signValue * conj(a(index));
    end
end

function local_validate_algebra(alg)
    isValid = isstruct(alg) && isscalar(alg) && isfield(alg, 'm') && ...
        isfield(alg, 'nBlades') && isfield(alg, 'starBlade') && ...
        isa(alg.starBlade, 'function_handle') && ...
        isnumeric(alg.m) && isreal(alg.m) && isscalar(alg.m) && ...
        isfinite(alg.m) && alg.m == floor(alg.m) && ...
        alg.m >= 0 && alg.m <= 52 && ...
        isnumeric(alg.nBlades) && isreal(alg.nBlades) && ...
        isscalar(alg.nBlades) && isfinite(alg.nBlades) && ...
        alg.nBlades == 2^alg.m;

    if ~isValid
        error('clifford_multivector_star:InvalidAlgebra', ...
            ['ALG must be a structure returned by clifford_algebra, with ', ...
             'a valid starBlade function handle.']);
    end
end

function coefficientVector = local_validate_multivector(value, nBlades)
    if ~(isnumeric(value) && isvector(value) && numel(value) == nBlades)
        error('clifford_multivector_star:InvalidCoefficientVector', ...
            'A must be a numeric vector with exactly ALG.nBlades entries.');
    end

    coefficientVector = value(:);
    if ~all(isfinite(coefficientVector))
        error('clifford_multivector_star:NonFiniteCoefficient', ...
            'A contains a NaN or Inf coefficient.');
    end
end
