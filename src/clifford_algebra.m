function alg = clifford_algebra(m)
%CLIFFORD_ALGEBRA Metadata and blade operations for the algebra Cl_{0,m}.
%
%   ALG = CLIFFORD_ALGEBRA(M) constructs a small, numeric description of the
%   real Clifford algebra Cl_{0,m}, with convention
%
%       e_i^2 = -1,              e_i e_j = -e_j e_i  (i ~= j).
%
%   No dimension-specific basis names are stored.  A basis blade is encoded by
%   a uint64 bit mask: bit i-1 is one exactly when e_i occurs in the blade.
%   Thus mask 0 is the scalar blade 1, mask 1 is e_1, mask 2 is e_2,
%   mask 3 is e_1*e_2, and so on.  If coefficient arrays are indexed in
%   MATLAB, blade mask MASK has index double(MASK)+1.
%
%   The returned structure contains function handles:
%
%       [s, c] = alg.multiplyMasks(a,b)
%           computes e_a e_b = s e_c, where a,b,c are uint64 masks.
%
%       [s, k] = alg.multiplyIndices(i,j)
%           does the same with one-based MATLAB basis indices.
%
%       r = alg.grade(mask)
%       label = alg.label(mask)
%       [s, mask] = alg.gradeInvolution(mask)
%       [s, mask] = alg.reversion(mask)
%       [s, mask] = alg.cliffordConjugation(mask)
%       [s, mask] = alg.starBlade(mask)  % alias for Clifford conjugation
%
%   starBlade acts on one real basis blade only.  A later multivector-level
%   star operation must also apply complex conjugation to its coefficients.
%
%   This constructor deliberately does not build a 2^m-by-2^m multiplication
%   table.  Products are evaluated on demand, which keeps the algebraic part
%   usable for arbitrary m within this package's safe MATLAB indexing limit
%   m <= 52.  Full
%   coefficient vectors and regular representations still grow exponentially
%   with m and should only be used for moderate dimensions.

    if ~(isnumeric(m) && isreal(m) && isscalar(m) && isfinite(m) && ...
            m == floor(m) && m >= 0 && m <= 52)
        error('clifford_algebra:InvalidDimension', ...
            'm must be an integer between 0 and 52.');
    end

    m = double(m);

    alg = struct();
    alg.name = sprintf('Cl_{0,%d}', m);
    alg.m = m;
    alg.signature = -ones(1, m);
    alg.nBlades = 2^m;
    alg.scalarMask = uint64(0);
    alg.fullMask = bitshift(uint64(1), m) - uint64(1);
    if m == 0
        alg.unitMasks = zeros(1, 0, 'uint64');
    else
        alg.unitMasks = bitshift(uint64(1), 0:(m - 1));
    end

    % Masks are zero-based; coefficient-array indices are one-based.
    alg.maskToIndex = @(mask) local_mask_to_index(mask, m);
    alg.indexToMask = @(index) local_index_to_mask(index, m);
    alg.grade = @(mask) local_grade(mask, m);
    alg.isEven = @(mask) mod(local_grade(mask, m), 2) == 0;
    alg.label = @(mask) local_label(mask, m);

    alg.multiplyMasks = @(leftMask, rightMask) ...
        local_multiply_masks(leftMask, rightMask, m);
    alg.multiplyIndices = @(leftIndex, rightIndex) ...
        local_multiply_indices(leftIndex, rightIndex, m);

    alg.gradeInvolution = @(mask) local_grade_involution(mask, m);
    alg.reversion = @(mask) local_reversion(mask, m);
    alg.cliffordConjugation = @(mask) ...
        local_clifford_conjugation(mask, m);
    alg.starBlade = alg.cliffordConjugation;
end

function [signValue, productMask] = local_multiply_masks(leftMask, rightMask, m)
% Compute e_leftMask * e_rightMask in the increasing-index blade basis.

    leftMask = local_validate_mask(leftMask, m);
    rightMask = local_validate_mask(rightMask, m);

    % A generator from the right blade swaps past every larger generator in
    % the left blade.  Each common generator additionally contributes e_i^2=-1.
    swaps = 0;
    for bit = 1:m
        if bitget(rightMask, bit)
            swaps = swaps + local_popcount(bitshift(leftMask, -bit));
        end
    end
    repeated = local_popcount(bitand(leftMask, rightMask));

    if mod(swaps + repeated, 2) == 0
        signValue = 1;
    else
        signValue = -1;
    end
    productMask = bitxor(leftMask, rightMask);
end

function [signValue, productIndex] = local_multiply_indices(leftIndex, rightIndex, m)
% Same product, with one-based MATLAB coefficient-array indices.

    leftMask = local_index_to_mask(leftIndex, m);
    rightMask = local_index_to_mask(rightIndex, m);
    [signValue, productMask] = local_multiply_masks(leftMask, rightMask, m);
    productIndex = double(productMask) + 1;
end

function index = local_mask_to_index(mask, m)
    mask = local_validate_mask(mask, m);
    index = double(mask) + 1;
end

function mask = local_index_to_mask(index, m)
    nBlades = 2^m;
    if ~(isnumeric(index) && isreal(index) && isscalar(index) && ...
            isfinite(index) && index == floor(index) && ...
            index >= 1 && index <= nBlades)
        error('clifford_algebra:InvalidIndex', ...
            'A blade index must be an integer between 1 and 2^m.');
    end
    mask = uint64(index - 1);
end

function mask = local_validate_mask(mask, m)
    if ~(isnumeric(mask) && isreal(mask) && isscalar(mask))
        error('clifford_algebra:InvalidMask', ...
            'A blade mask must be one real numeric scalar.');
    end

    if isa(mask, 'uint64')
        numericMask = mask;
    else
        if ~(isfinite(mask) && mask == floor(mask) && mask >= 0)
            error('clifford_algebra:InvalidMask', ...
                'A blade mask must be a nonnegative integer.');
        end
        numericMask = uint64(mask);
    end

    fullMask = bitshift(uint64(1), m) - uint64(1);
    if numericMask > fullMask
        error('clifford_algebra:InvalidMask', ...
            'The mask contains a generator outside Cl_{0,m}.');
    end
    mask = numericMask;
end

function r = local_grade(mask, m)
    mask = local_validate_mask(mask, m);
    r = local_popcount(mask);
end

function count = local_popcount(mask)
% Kernighan's bit-count algorithm for one uint64 scalar.

    count = 0;
    while mask ~= uint64(0)
        mask = bitand(mask, mask - uint64(1));
        count = count + 1;
    end
end

function label = local_label(mask, m)
    mask = local_validate_mask(mask, m);
    if mask == uint64(0)
        label = '1';
        return;
    end

    factors = cell(1, local_popcount(mask));
    position = 1;
    for bit = 1:m
        if bitget(mask, bit)
            factors{position} = sprintf('e%d', bit);
            position = position + 1;
        end
    end
    label = strjoin(factors, '*');
end

function [signValue, outputMask] = local_grade_involution(mask, m)
    outputMask = local_validate_mask(mask, m);
    if mod(local_popcount(outputMask), 2) == 0
        signValue = 1;
    else
        signValue = -1;
    end
end

function [signValue, outputMask] = local_reversion(mask, m)
    outputMask = local_validate_mask(mask, m);
    r = local_popcount(outputMask);
    if mod(r * (r - 1) / 2, 2) == 0
        signValue = 1;
    else
        signValue = -1;
    end
end

function [signValue, outputMask] = local_clifford_conjugation(mask, m)
    outputMask = local_validate_mask(mask, m);
    r = local_popcount(outputMask);
    if mod(r * (r + 1) / 2, 2) == 0
        signValue = 1;
    else
        signValue = -1;
    end
end
