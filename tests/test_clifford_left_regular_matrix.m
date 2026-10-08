function results = test_clifford_left_regular_matrix(dimensions)
%TEST_CLIFFORD_LEFT_REGULAR_MATRIX Test the faithful left-regular map.
%
%   RESULTS = TEST_CLIFFORD_LEFT_REGULAR_MATRIX() tests m=4 and m=5.  Its
%   reference construction uses signed generator matrices directly from
%   bit masks; it does not call ALG.multiplyIndices or the product routine.

    if nargin == 0
        dimensions = [4, 5];
    end

    if ~(isnumeric(dimensions) && isreal(dimensions) && isvector(dimensions) && ...
            all(isfinite(dimensions)) && all(dimensions == floor(dimensions)) && ...
            all(dimensions >= 0))
        error('test_clifford_left_regular_matrix:InvalidDimensions', ...
            'DIMENSIONS must be a vector of nonnegative integer dimensions.');
    end

    local_edge_and_input_checks();
    savedRandomState = rng;
    restoreRandomState = onCleanup(@() rng(savedRandomState)); %#ok<NASGU>

    template = struct('m', [], 'nBlades', [], 'referenceResidual', [], ...
        'actionResidual', [], 'homomorphismResidual', [], ...
        'starAdjointResidual', [], 'decoderResidual', []);
    results = repmat(template, 1, numel(dimensions));

    for testNumber = 1:numel(dimensions)
        m = dimensions(testNumber);
        alg = clifford_algebra(m);
        nBlades = alg.nBlades;
        generatorMatrices = local_generator_matrices(m);
        bladeMatrices = local_blade_matrices(generatorMatrices, m);

        rng(3000 + m, 'twister');
        a = randn(nBlades, 1) + 1i * randn(nBlades, 1);
        b = randn(nBlades, 1) + 1i * randn(nBlades, 1);

        leftRegularA = clifford_left_regular_matrix(a, alg);
        leftRegularB = clifford_left_regular_matrix(b, alg);
        referenceA = local_reference_matrix(a, bladeMatrices);
        referenceResidual = norm(leftRegularA - referenceA, inf);
        local_assert_close(leftRegularA, referenceA);
        assert(isequal(size(leftRegularA), [nBlades, nBlades]));
        local_assert_close(clifford_left_regular_matrix(a.', alg), leftRegularA);
        assert(isequal(leftRegularA(:, 1), a));
        local_basis_matrix_checks(alg, bladeMatrices);

        % The matrix acts exactly as left multiplication on coefficients.
        action = leftRegularA * b;
        expectedAction = clifford_multivector_product(a, b, alg);
        actionResidual = norm(action - expectedAction, inf);
        local_assert_close(action, expectedAction);

        % It is a complex-linear algebra homomorphism.
        alpha = 2 - 3i;
        beta = -1 + 4i;
        local_assert_close(clifford_left_regular_matrix(alpha * a + beta * b, alg), ...
            alpha * leftRegularA + beta * leftRegularB);
        ab = clifford_multivector_product(a, b, alg);
        leftRegularAB = clifford_left_regular_matrix(ab, alg);
        homomorphismResidual = norm(leftRegularAB - leftRegularA * leftRegularB, inf);
        local_assert_close(leftRegularAB, leftRegularA * leftRegularB);
        local_basis_pair_homomorphism_checks(alg);

        % The Clifford star becomes the ordinary complex matrix adjoint.
        aStar = clifford_multivector_star(a, alg);
        leftRegularStar = clifford_left_regular_matrix(aStar, alg);
        starAdjointResidual = norm(leftRegularStar - leftRegularA', inf);
        local_assert_close(leftRegularStar, leftRegularA');
        starProduct = clifford_multivector_product(aStar, a, alg);
        local_assert_close(leftRegularA' * leftRegularA, ...
            clifford_left_regular_matrix(starProduct, alg));

        % The decoder is a genuine inverse on the left-regular image.
        [recoveredA, decoderResidual] = ...
            clifford_multivector_from_left_regular_matrix(leftRegularA, alg);
        local_assert_close(recoveredA, a);
        local_assert_close(clifford_left_regular_matrix(recoveredA, alg), ...
            leftRegularA);

        one = zeros(nBlades, 1); one(1) = 1;
        zero = zeros(nBlades, 1);
        assert(isequal(clifford_left_regular_matrix(one, alg), eye(nBlades)));
        assert(isequal(clifford_left_regular_matrix(zero, alg), zeros(nBlades)));
        local_assert_close(trace(leftRegularA), nBlades * a(1));
        local_assert_close(norm(leftRegularA, 'fro')^2, ...
            nBlades * sum(abs(a).^2));

        results(testNumber).m = m;
        results(testNumber).nBlades = nBlades;
        results(testNumber).referenceResidual = referenceResidual;
        results(testNumber).actionResidual = actionResidual;
        results(testNumber).homomorphismResidual = homomorphismResidual;
        results(testNumber).starAdjointResidual = starAdjointResidual;
        results(testNumber).decoderResidual = decoderResidual;
        fprintf(['Cl_{0,%d}: %d-by-%d left-regular matrix; ', ...
            'faithful representation passed.\n'], m, nBlades, nBlades);
    end
end

function local_basis_matrix_checks(alg, bladeMatrices)
    nBlades = alg.nBlades;
    for bladeIndex = 1:nBlades
        blade = zeros(nBlades, 1);
        blade(bladeIndex) = 1;
        actual = clifford_left_regular_matrix(blade, alg);
        expected = full(bladeMatrices{bladeIndex});
        assert(isequal(actual, expected));
    end
end

function local_basis_pair_homomorphism_checks(alg)
    nBlades = alg.nBlades;
    for leftIndex = 1:nBlades
        leftBlade = zeros(nBlades, 1);
        leftBlade(leftIndex) = 1;
        leftMatrix = clifford_left_regular_matrix(leftBlade, alg);
        for rightIndex = 1:nBlades
            rightBlade = zeros(nBlades, 1);
            rightBlade(rightIndex) = 1;
            productBlade = clifford_multivector_product(leftBlade, rightBlade, alg);
            actual = clifford_left_regular_matrix(productBlade, alg);
            expected = leftMatrix * clifford_left_regular_matrix(rightBlade, alg);
            assert(isequal(actual, expected));
        end
    end
end

function generatorMatrices = local_generator_matrices(m)
% Direct signed-permutation realization of e_i acting on the left.

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

function bladeMatrices = local_blade_matrices(generatorMatrices, m)
    nBlades = 2^m;
    bladeMatrices = cell(nBlades, 1);
    for bladeIndex = 1:nBlades
        bladeMask = uint64(bladeIndex - 1);
        bladeMatrix = speye(nBlades);
        for generator = 1:m
            if bitget(bladeMask, generator)
                bladeMatrix = bladeMatrix * generatorMatrices{generator};
            end
        end
        bladeMatrices{bladeIndex} = bladeMatrix;
    end
end

function reference = local_reference_matrix(a, bladeMatrices)
    nBlades = numel(a);
    reference = zeros(nBlades, nBlades);
    for bladeIndex = 1:nBlades
        if a(bladeIndex) ~= 0
            reference = reference + a(bladeIndex) * ...
                full(bladeMatrices{bladeIndex});
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
    dimensionFactor = max(1, sqrt(numel(actual)));
    assert(residual <= 5000 * dimensionFactor * eps(scale));
end

function local_edge_and_input_checks()
    alg0 = clifford_algebra(0);
    scalar = 2 - 3i;
    assert(isequal(clifford_left_regular_matrix(scalar, alg0), scalar));
    [recoveredScalar, scalarResidual] = ...
        clifford_multivector_from_left_regular_matrix(scalar, alg0);
    assert(isequal(recoveredScalar, scalar));
    assert(scalarResidual == 0);

    alg1 = clifford_algebra(1);
    e1 = [0; 1];
    expectedE1 = [0, -1; 1, 0];
    assert(isequal(clifford_left_regular_matrix(e1, alg1), expectedE1));

    alg2 = clifford_algebra(2);
    e1 = [0; 1; 0; 0];
    e2 = [0; 0; 1; 0];
    leftE1 = clifford_left_regular_matrix(e1, alg2);
    leftE2 = clifford_left_regular_matrix(e2, alg2);
    assert(isequal(leftE1 * leftE1, -eye(4)));
    assert(isequal(leftE2 * leftE2, -eye(4)));
    assert(isequal(leftE1 * leftE2, -leftE2 * leftE1));

    local_assert_error(@() clifford_left_regular_matrix([1; 2], alg2), ...
        'clifford_left_regular_matrix:InvalidCoefficientVector');
    local_assert_error(@() clifford_left_regular_matrix(ones(4), alg2));
    local_assert_error(@() clifford_left_regular_matrix([1; 0; NaN; 0], alg2));
    local_assert_error(@() clifford_left_regular_matrix([1; 0; Inf; 0], alg2));
    local_assert_error(@() clifford_left_regular_matrix('abcd', alg2));
    local_assert_error(@() clifford_left_regular_matrix(zeros(4, 1), struct()));

    local_assert_error(@() ...
        clifford_multivector_from_left_regular_matrix(ones(3, 4), alg2));
    local_assert_error(@() clifford_multivector_from_left_regular_matrix( ...
        [NaN, 0, 0, 0; zeros(3, 4)], alg2));
    local_assert_error(@() clifford_multivector_from_left_regular_matrix( ...
        eye(4), alg2, -1));
    local_assert_error(@() clifford_multivector_from_left_regular_matrix( ...
        eye(4), alg2, 1i));

    notLeftRegular = eye(4);
    notLeftRegular(1, 2) = 0.1;
    local_assert_error(@() clifford_multivector_from_left_regular_matrix( ...
        notLeftRegular, alg2), ...
        'clifford_multivector_from_left_regular_matrix:NotInLeftRegularImage');

    slightlyPerturbed = eye(4);
    slightlyPerturbed(1, 2) = 1e-8;
    [recoveredOne, perturbedResidual] = ...
        clifford_multivector_from_left_regular_matrix( ...
            slightlyPerturbed, alg2, 1e-6);
    assert(isequal(recoveredOne, [1; 0; 0; 0]));
    assert(perturbedResidual > 0 && perturbedResidual < 1e-6);
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
