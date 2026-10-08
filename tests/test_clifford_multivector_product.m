function results = test_clifford_multivector_product(dimensions)
%TEST_CLIFFORD_MULTIVECTOR_PRODUCT Test general complex Clifford products.
%
%   RESULTS = TEST_CLIFFORD_MULTIVECTOR_PRODUCT() tests dimensions m=4 and
%   m=5.  The test uses an independent generator-matrix realization of
%   Cl_{0,m}; it does not use ALG.multiplyIndices for its reference answer.

    if nargin == 0
        dimensions = [4, 5];
    end

    if ~(isnumeric(dimensions) && isreal(dimensions) && isvector(dimensions))
        error('test_clifford_multivector_product:InvalidDimensions', ...
            'DIMENSIONS must be a real numeric vector.');
    end

    local_input_validation_checks();
    savedRandomState = rng;
    restoreRandomState = onCleanup(@() rng(savedRandomState)); %#ok<NASGU>

    template = struct('m', [], 'nBlades', [], ...
        'leftRegularResidual', [], 'regularRepresentationResidual', [], ...
        'associativityResidual', []);
    results = repmat(template, 1, numel(dimensions));

    for testNumber = 1:numel(dimensions)
        m = dimensions(testNumber);
        alg = clifford_algebra(m);
        nBlades = alg.nBlades;
        generatorMatrices = local_generator_matrices(m);
        bladeLeftMatrices = local_blade_left_matrices(generatorMatrices, m);

        local_basis_checks(alg, bladeLeftMatrices);

        % Deterministic complex data make failures reproducible.
        rng(1000 + m, 'twister');
        a = randn(nBlades, 1) + 1i * randn(nBlades, 1);
        b = randn(nBlades, 1) + 1i * randn(nBlades, 1);
        c = randn(nBlades, 1) + 1i * randn(nBlades, 1);
        d = randn(nBlades, 1) + 1i * randn(nBlades, 1);

        product = clifford_multivector_product(a, b, alg);
        leftRegularA = local_left_regular_matrix(a, bladeLeftMatrices);
        leftRegularB = local_left_regular_matrix(b, bladeLeftMatrices);
        leftRegularProduct = leftRegularA * b;
        leftRegularResidual = norm(product - leftRegularProduct, inf);
        local_assert_close(product, leftRegularProduct);
        assert(isequal(size(product), [nBlades, 1]));
        local_assert_close(leftRegularA(:, 1), a);

        % Faithfulness of the independently constructed left-regular action.
        leftRegularProductMatrix = local_left_regular_matrix( ...
            product, bladeLeftMatrices);
        regularRepresentationResidual = norm( ...
            leftRegularA * leftRegularB - leftRegularProductMatrix, inf);
        local_assert_close(leftRegularA * leftRegularB, ...
            leftRegularProductMatrix);

        % Bilinearity over complex scalars.
        alpha = 2 - 3i;
        beta = -1 + 4i;
        linearLeft = clifford_multivector_product(alpha * a + beta * d, c, alg);
        expectedLeft = alpha * clifford_multivector_product(a, c, alg) + ...
            beta * clifford_multivector_product(d, c, alg);
        local_assert_close(linearLeft, expectedLeft);

        linearRight = clifford_multivector_product(a, alpha * b + beta * d, alg);
        expectedRight = alpha * clifford_multivector_product(a, b, alg) + ...
            beta * clifford_multivector_product(a, d, alg);
        local_assert_close(linearRight, expectedRight);
        local_assert_close(clifford_multivector_product(a.', b, alg), product);
        local_assert_close(clifford_multivector_product(a, b.', alg), product);

        % Associativity is checked on general complex multivectors.
        leftAssociated = clifford_multivector_product( ...
            clifford_multivector_product(a, b, alg), c, alg);
        rightAssociated = clifford_multivector_product( ...
            a, clifford_multivector_product(b, c, alg), alg);
        associativityResidual = norm(leftAssociated - rightAssociated, inf);
        local_assert_close(leftAssociated, rightAssociated);

        % The scalar unit, zero vector, and central complex scalars behave as expected.
        one = zeros(nBlades, 1); one(1) = 1;
        zero = zeros(nBlades, 1);
        assert(norm(clifford_multivector_product(one, a, alg) - a, inf) == 0);
        assert(norm(clifford_multivector_product(a, one, alg) - a, inf) == 0);
        assert(norm(clifford_multivector_product(a, zero, alg), inf) == 0);
        complexScalar = alpha * one;
        local_assert_close(clifford_multivector_product(complexScalar, a, alg), ...
            alpha * a);
        local_assert_close(clifford_multivector_product(a, complexScalar, alg), ...
            alpha * a);

        results(testNumber).m = m;
        results(testNumber).nBlades = nBlades;
        results(testNumber).leftRegularResidual = leftRegularResidual;
        results(testNumber).regularRepresentationResidual = ...
            regularRepresentationResidual;
        results(testNumber).associativityResidual = associativityResidual;
        fprintf(['Cl_{0,%d}: %d coefficient components; ', ...
            'complex multivector product passed.\n'], m, nBlades);
    end
end

function local_basis_checks(alg, bladeLeftMatrices)
    nBlades = alg.nBlades;

    if alg.m >= 1
        e1 = zeros(nBlades, 1); e1(2) = 1;
        assert(norm(clifford_multivector_product(e1, e1, alg) + ...
            local_scalar_one(nBlades), inf) == 0);
    end
    if alg.m >= 2
        e1 = zeros(nBlades, 1); e1(2) = 1;
        e2 = zeros(nBlades, 1); e2(3) = 1;
        e12 = zeros(nBlades, 1); e12(4) = 1;
        assert(norm(clifford_multivector_product(e1, e2, alg) - e12, inf) == 0);
        assert(norm(clifford_multivector_product(e2, e1, alg) + e12, inf) == 0);
    end

    % Check every pair of basis blades against an independent construction.
    for leftIndex = 1:nBlades
        leftBlade = zeros(nBlades, 1);
        leftBlade(leftIndex) = 1;
        for rightIndex = 1:nBlades
            rightBlade = zeros(nBlades, 1);
            rightBlade(rightIndex) = 1;
            actual = clifford_multivector_product(leftBlade, rightBlade, alg);
            expected = full(bladeLeftMatrices{leftIndex} * rightBlade);
            assert(isequal(actual, expected));
        end
    end
end

function one = local_scalar_one(nBlades)
    one = zeros(nBlades, 1);
    one(1) = 1;
end

function generatorMatrices = local_generator_matrices(m)
% Build the left action of each e_i directly from the Clifford relations.

    nBlades = 2^m;
    generatorMatrices = cell(1, m);

    for generator = 1:m
        generatorMatrix = sparse(nBlades, nBlades);
        generatorMask = bitshift(uint64(1), generator - 1);
        lowerMask = generatorMask - uint64(1);

        for column = 1:nBlades
            bladeMask = uint64(column - 1);
            lowerCount = local_popcount(bitand(bladeMask, lowerMask));
            repeated = double(bitget(bladeMask, generator));
            signValue = (-1)^(lowerCount + repeated);
            outputIndex = double(bitxor(bladeMask, generatorMask)) + 1;
            generatorMatrix(outputIndex, column) = signValue;
        end
        generatorMatrices{generator} = generatorMatrix;
    end
end

function bladeLeftMatrices = local_blade_left_matrices(generatorMatrices, m)
% Construct L_{e_I}=E_{i_1}...E_{i_r} for canonical increasing blades.

    nBlades = 2^m;
    bladeLeftMatrices = cell(nBlades, 1);
    for bladeIndex = 1:nBlades
        bladeMask = uint64(bladeIndex - 1);
        bladeMatrix = speye(nBlades);
        for generator = 1:m
            if bitget(bladeMask, generator)
                bladeMatrix = bladeMatrix * generatorMatrices{generator};
            end
        end
        bladeLeftMatrices{bladeIndex} = bladeMatrix;
    end
end

function leftRegular = local_left_regular_matrix(a, bladeLeftMatrices)
% L_a = sum_I a_I L_{e_I}, using the independent blade matrices above.

    nBlades = numel(a);
    leftRegular = sparse(nBlades, nBlades);
    for bladeIndex = 1:nBlades
        coefficient = a(bladeIndex);
        if coefficient ~= 0
            leftRegular = leftRegular + ...
                coefficient * bladeLeftMatrices{bladeIndex};
        end
    end
end

function count = local_popcount(mask)
    count = 0;
    while mask ~= uint64(0)
        mask = bitand(mask, mask - uint64(1));
        count = count + 1;
    end
end

function local_assert_close(actual, expected)
    scale = max([1; abs(actual(:)); abs(expected(:))]);
    residual = norm(actual(:) - expected(:), inf);
    assert(residual <= 5000 * eps(scale));
end

function local_input_validation_checks()
    alg0 = clifford_algebra(0);
    scalarProduct = clifford_multivector_product(2 + 3i, -1 + 4i, alg0);
    assert(isequal(scalarProduct, (2 + 3i) * (-1 + 4i)));

    alg1 = clifford_algebra(1);
    e1 = [0; 1];
    assert(isequal(clifford_multivector_product(e1, e1, alg1), [-1; 0]));

    alg = clifford_algebra(2);
    local_assert_error(@() clifford_multivector_product([1; 2], [1; 2], alg), ...
        'clifford_multivector_product:InvalidLeftCoefficientVector');
    local_assert_error(@() clifford_multivector_product( ...
        zeros(4, 1), ones(5, 1), alg), ...
        'clifford_multivector_product:InvalidRightCoefficientVector');
    local_assert_error(@() clifford_multivector_product( ...
        ones(4), ones(4, 1), alg));
    local_assert_error(@() clifford_multivector_product( ...
        [1; 0; NaN; 0], zeros(4, 1), alg));
    local_assert_error(@() clifford_multivector_product( ...
        zeros(4, 1), [1; 0; Inf; 0], alg));
    local_assert_error(@() clifford_multivector_product('abcd', zeros(4, 1), alg));
    local_assert_error(@() clifford_multivector_product( ...
        zeros(4, 1), zeros(4, 1), struct()));
end

function local_assert_error(action, expectedId)
    if nargin < 2
        expectedId = '';
    end
    didError = false;
    try
        action();
    catch exception
        didError = true;
        if ~isempty(expectedId)
            assert(strcmp(exception.identifier, expectedId));
        end
    end
    assert(didError);
end
