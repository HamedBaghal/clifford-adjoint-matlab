function results = test_clifford_spherical_monogenic_gram(dimensions)
%TEST_CLIFFORD_SPHERICAL_MONOGENIC_GRAM Test kernel, Gram, and block map.
%
%   RESULTS = TEST_CLIFFORD_SPHERICAL_MONOGENIC_GRAM() tests m=2,3,4,5.
%   The tests use closed low-degree kernel formulas independently of the
%   implementation recurrence, then bridge the tensor to the existing block
%   map, decoder, and Hermitian eigensystem utilities.

    if nargin == 0
        dimensions = [2, 3, 4, 5];
    end
    if ~(isnumeric(dimensions) && isreal(dimensions) && isvector(dimensions) && ...
            all(isfinite(dimensions)) && all(dimensions == floor(dimensions)) && ...
            all(dimensions >= 2))
        error('test_clifford_spherical_monogenic_gram:InvalidDimensions', ...
            'DIMENSIONS must be a vector of integer dimensions at least 2.');
    end

    savedRandomState = rng;
    restoreRandomState = onCleanup(@() rng(savedRandomState)); %#ok<NASGU>
    local_edge_and_option_checks();
    local_circle_limit_checks();
    local_degree_two_closed_form_checks();

    template = struct('m', [], 'nBlades', [], 'diagonalLevel', [], ...
        'lambdaMin', [], 'lambdaMax', [], 'conditionNumber2', [], ...
        'objectiveF', []);
    results = repmat(template, 1, numel(dimensions));

    for testNumber = 1:numel(dimensions)
        m = dimensions(testNumber);
        alg = clifford_algebra(m);
        nBlades = alg.nBlades;
        points = local_two_coordinate_points(m);

        % K_0 is identically one, independently of dimension and points.
        [zeroEntries, zeroInfo] = clifford_spherical_monogenic_gram( ...
            points, 0, alg);
        expectedZero = complex(zeros(2, 2, nBlades));
        expectedZero(:, :, 1) = ones(2);
        local_assert_close(zeroEntries, expectedZero);
        assert(zeroInfo.diagonalLevel == 1);
        local_assert_close(zeroInfo.objectiveF, 1);
        [zeroBlock, zeroMapInfo] = ...
            clifford_matrix_left_regular_matrix(zeroEntries, alg);
        local_assert_close(zeroBlock, kron(ones(2), eye(nBlades)));
        assert(zeroMapInfo.isSquare);
        zeroSpectrum = clifford_matrix_left_regular_eigensystem( ...
            zeroEntries, alg);
        assert(zeroSpectrum.hermitian.isAvailable);
        local_assert_same_real_values(zeroSpectrum.hermitian.eigenvalues, ...
            [zeros(nBlades, 1); 2 * ones(nBlades, 1)]);
        assert(~zeroSpectrum.hermitian.isNumericallyPositiveDefinite);

        % Universal degree-one formula: K_1(x,y)=(m-1)<x,y>+x wedge y.
        [degreeOneEntries, degreeOneInfo] = ...
            clifford_spherical_monogenic_gram(points, 1, alg);
        expectedDegreeOne = local_degree_one_oracle(points, alg);
        local_assert_close(degreeOneEntries, expectedDegreeOne);
        assert(degreeOneInfo.diagonalLevel == m - 1);
        assert(degreeOneInfo.monogenicModuleRank == m - 1);
        assert(degreeOneInfo.isSquareInterpolationEnsemble == (m == 3));
        local_assert_close(degreeOneInfo.objectiveF, 1);
        local_assert_star_hermitian(degreeOneEntries, alg);
        local_assert_only_scalar_and_bivector_grades(degreeOneEntries, alg);

        [degreeOneBlock, degreeOneMapInfo] = ...
            clifford_matrix_left_regular_matrix(degreeOneEntries, alg);
        assert(degreeOneMapInfo.isSquare);
        local_assert_close(degreeOneBlock, degreeOneBlock');
        local_assert_close(norm(degreeOneBlock, 'fro')^2, ...
            nBlades * degreeOneInfo.totalCoefficientFrobeniusEnergy);
        offDiagonal = degreeOneBlock - ...
            degreeOneInfo.diagonalLevel * eye(2 * nBlades);
        local_assert_close(norm(offDiagonal, 'fro')^2, ...
            degreeOneInfo.fullRegularOffDiagonalFrobeniusEnergy);

        [decodedEntries, decoderInfo] = ...
            clifford_matrix_from_left_regular_matrix(degreeOneBlock, alg);
        local_assert_close(decodedEntries, degreeOneEntries);
        assert(decoderInfo.isInBlockLeftRegularImage);

        degreeOneSpectrum = clifford_matrix_left_regular_eigensystem( ...
            degreeOneEntries, alg);
        hermitian = degreeOneSpectrum.hermitian;
        assert(degreeOneSpectrum.isNumericallyStarSelfAdjoint);
        assert(hermitian.isAvailable);
        expectedValues = [(m - 2) * ones(nBlades, 1); ...
            m * ones(nBlades, 1)];
        local_assert_same_real_values(hermitian.eigenvalues, expectedValues);
        assert(hermitian.lambdaMin >= -hermitian.positiveDefiniteTolerance);
        if m == 2
            assert(~hermitian.isNumericallyPositiveDefinite);
        else
            assert(hermitian.isNumericallyPositiveDefinite);
            local_assert_close(hermitian.positiveDefiniteConditionNumber2, ...
                m / (m - 2));
        end

        % K_1(x,-x)=-(m-1), with no bivector part.
        x = points(1, :);
        [antipodalKernel, antipodalInfo] = ...
            clifford_spherical_monogenic_kernel(x, -x, 1, alg);
        expectedAntipodal = complex(zeros(nBlades, 1));
        expectedAntipodal(1) = -(m - 1);
        local_assert_close(antipodalKernel, expectedAntipodal);
        local_assert_close(antipodalInfo.diagonalLevel, m - 1);

        % Permuting centres produces a block permutation similarity.
        permutation = [2, 1];
        permutedEntries = clifford_spherical_monogenic_gram( ...
            points(permutation, :), 1, alg);
        [permutedBlock, ~] = clifford_matrix_left_regular_matrix( ...
            permutedEntries, alg);
        P = eye(2);
        P = P(permutation, :);
        blockPermutation = kron(P, eye(nBlades));
        local_assert_close(permutedBlock, ...
            blockPermutation * degreeOneBlock * blockPermutation');

        results(testNumber).m = m;
        results(testNumber).nBlades = nBlades;
        results(testNumber).diagonalLevel = degreeOneInfo.diagonalLevel;
        results(testNumber).lambdaMin = hermitian.lambdaMin;
        results(testNumber).lambdaMax = hermitian.lambdaMax;
        results(testNumber).conditionNumber2 = ...
            hermitian.positiveDefiniteConditionNumber2;
        results(testNumber).objectiveF = degreeOneInfo.objectiveF;
        fprintf(['Cl_{0,%d}: spherical-monogenic kernel, Gram matrix, ', ...
            'and block diagnostics passed.\n'], m);
    end
end

function points = local_two_coordinate_points(m)
    points = zeros(2, m);
    points(1, 1) = 1;
    points(2, 2) = 1;
end

function entries = local_degree_one_oracle(points, alg)
    pointCount = size(points, 1);
    entries = complex(zeros(pointCount, pointCount, alg.nBlades));
    for row = 1:pointCount
        for column = 1:pointCount
            coefficient = complex(zeros(alg.nBlades, 1));
            coefficient(1) = (alg.m - 1) * ...
                (points(row, :) * points(column, :).');
            for firstCoordinate = 1:alg.m
                for secondCoordinate = (firstCoordinate + 1):alg.m
                    mask = bitset(bitset(uint64(0), firstCoordinate, 1), ...
                        secondCoordinate, 1);
                    index = alg.maskToIndex(mask);
                    coefficient(index) = points(row, firstCoordinate) * ...
                        points(column, secondCoordinate) - ...
                        points(row, secondCoordinate) * ...
                        points(column, firstCoordinate);
                end
            end
            entries(row, column, :) = reshape( ...
                coefficient, [1, 1, alg.nBlades]);
        end
    end
end

function local_circle_limit_checks()
    alg = clifford_algebra(2);
    theta = [0; pi / 6; pi / 2; pi];
    points = [cos(theta), sin(theta)];
    nBlades = alg.nBlades;
    e12Index = alg.maskToIndex(uint64(3));

    for degree = [0, 1, 2, 4]
        [entries, info] = clifford_spherical_monogenic_gram(points, degree, alg);
        expected = complex(zeros(4, 4, nBlades));
        for row = 1:4
            for column = 1:4
                if degree == 0
                    expected(row, column, 1) = 1;
                else
                    difference = theta(row) - theta(column);
                    expected(row, column, 1) = cos(degree * difference);
                    expected(row, column, e12Index) = ...
                        -sin(degree * difference);
                end
            end
        end
        local_assert_close(entries, expected);
        assert(info.usesTwoDimensionalLimit);
        assert(info.diagonalLevel == 1);
        local_assert_star_hermitian(entries, alg);

        [block, ~] = clifford_matrix_left_regular_matrix(entries, alg);
        spectrum = clifford_matrix_left_regular_eigensystem(entries, alg);
        assert(spectrum.hermitian.isAvailable);
        assert(spectrum.hermitian.lambdaMin >= ...
            -spectrum.hermitian.positiveDefiniteTolerance);
        assert(spectrum.numericalRank == nBlades);
        local_assert_close(norm(block - block', 'fro'), 0);
    end
end

function local_degree_two_closed_form_checks()
    for m = [3, 4, 5]
        alg = clifford_algebra(m);
        nBlades = alg.nBlades;
        x = zeros(1, m);
        x(1) = 1;
        y = zeros(1, m);
        y(1:2) = [1, 1] / sqrt(2);
        t = x * y.';
        [kernel, kernelInfo] = ...
            clifford_spherical_monogenic_kernel(x, y, 2, alg);
        expected = complex(zeros(nBlades, 1));
        expected(1) = (m / 2) * (m * t^2 - 1);
        expected(alg.maskToIndex(uint64(3))) = m * t^2;
        local_assert_close(kernel, expected);
        assert(kernelInfo.diagonalLevel == nchoosek(m, 2));

        % Three orthogonal centres have a scalar Gram matrix with known
        % packed eigenvalues.  This includes the earlier m=4,k=2 level 6.
        points = zeros(3, m);
        points(1, 1) = 1;
        points(2, 2) = 1;
        points(3, 3) = 1;
        [entries, info] = clifford_spherical_monogenic_gram(points, 2, alg);
        scalarLevel = m * (m - 1) / 2;
        offDiagonalLevel = -m / 2;
        expectedScalar = (scalarLevel - offDiagonalLevel) * eye(3) + ...
            offDiagonalLevel * ones(3);
        local_assert_close(entries(:, :, 1), expectedScalar);
        local_assert_close(entries(:, :, 2:end), zeros(3, 3, nBlades - 1));
        local_assert_close(info.objectiveF, 3 * offDiagonalLevel^2);
        [block, ~] = clifford_matrix_left_regular_matrix(entries, alg);
        spectrum = clifford_matrix_left_regular_eigensystem(entries, alg);
        expectedEigenvalues = [ ...
            (m * (m - 3) / 2) * ones(nBlades, 1); ...
            (m^2 / 2) * ones(2 * nBlades, 1)];
        local_assert_same_real_values(spectrum.hermitian.eigenvalues, ...
            expectedEigenvalues);
        if m == 3
            % This particular square interpolation ensemble is singular;
            % r=d is a natural square case, but is not sufficient for PD.
            assert(~spectrum.hermitian.isNumericallyPositiveDefinite);
        else
            assert(spectrum.hermitian.isNumericallyPositiveDefinite);
        end
        local_assert_close(spectrum.hermitian.lambdaMin, m * (m - 3) / 2);
        local_assert_close(norm(block - block', 'fro'), 0);

        if m == 4
            % Exact bridge to the previous four-dimensional F objective.
            mixedPoints = [x; y; local_unit_vector_sum(m, [1, 3])];
            [~, mixedInfo] = clifford_spherical_monogenic_gram( ...
                mixedPoints, 2, alg);
            expectedF = 0;
            for row = 1:3
                for column = (row + 1):3
                    innerProduct = mixedPoints(row, :) * mixedPoints(column, :).';
                    expectedF = expectedF + 48 * innerProduct^4 - ...
                        16 * innerProduct^2 + 4;
                end
            end
            local_assert_close(mixedInfo.objectiveF, expectedF);
        end
    end
end

function vector = local_unit_vector_sum(m, positions)
    vector = zeros(1, m);
    vector(positions) = 1;
    vector = vector / norm(vector);
end

function local_edge_and_option_checks()
    alg1 = clifford_algebra(1);
    local_assert_error(@() clifford_spherical_monogenic_gram( ...
        [1], 0, alg1), ...
        'clifford_spherical_monogenic_gram:UnsupportedDimension');
    local_assert_error(@() clifford_spherical_monogenic_kernel( ...
        [1], [1], 0, alg1), ...
        'clifford_spherical_monogenic_kernel:UnsupportedDimension');

    alg3 = clifford_algebra(3);
    points = local_two_coordinate_points(3);
    [baseline, baselineInfo] = clifford_spherical_monogenic_gram( ...
        points, 1, alg3);
    assert(~baselineInfo.normalizePoints);
    scaledPoints = 1.25 * points;
    local_assert_error(@() clifford_spherical_monogenic_gram( ...
        scaledPoints, 1, alg3), ...
        'clifford_spherical_monogenic_gram:NonUnitPoint');
    [normalized, normalizedInfo] = clifford_spherical_monogenic_gram( ...
        scaledPoints, 1, alg3, struct('normalizePoints', true));
    local_assert_close(normalized, baseline);
    assert(normalizedInfo.normalizePoints);
    local_assert_close(normalizedInfo.pointsUsed, points);

    local_assert_error(@() clifford_spherical_monogenic_gram( ...
        [], 1, alg3), ...
        'clifford_spherical_monogenic_gram:InvalidPoints');
    local_assert_error(@() clifford_spherical_monogenic_gram( ...
        zeros(2, 2), 1, alg3), ...
        'clifford_spherical_monogenic_gram:InvalidPoints');
    local_assert_error(@() clifford_spherical_monogenic_gram( ...
        [1, 0, 0; 0, 0, 0], 1, alg3), ...
        'clifford_spherical_monogenic_gram:ZeroPoint');
    local_assert_error(@() clifford_spherical_monogenic_gram( ...
        [1, 0, 0; 0, 1i, 0], 1, alg3), ...
        'clifford_spherical_monogenic_gram:InvalidPoints');
    badPoint = points;
    badPoint(1, 1) = NaN;
    local_assert_error(@() clifford_spherical_monogenic_gram( ...
        badPoint, 1, alg3), ...
        'clifford_spherical_monogenic_gram:InvalidPoints');
    local_assert_error(@() clifford_spherical_monogenic_gram( ...
        points, -1, alg3), ...
        'clifford_spherical_monogenic_gram:InvalidDegree');
    local_assert_error(@() clifford_spherical_monogenic_gram( ...
        points, 1.5, alg3), ...
        'clifford_spherical_monogenic_gram:InvalidDegree');
    local_assert_error(@() clifford_spherical_monogenic_gram( ...
        points, 1i, alg3), ...
        'clifford_spherical_monogenic_gram:InvalidDegree');
    local_assert_error(@() clifford_spherical_monogenic_gram( ...
        points, 1, alg3, 1), ...
        'clifford_spherical_monogenic_gram:InvalidOptions');
    local_assert_error(@() clifford_spherical_monogenic_gram( ...
        points, 1, alg3, struct('unknown', true)), ...
        'clifford_spherical_monogenic_gram:UnknownOption');
    local_assert_error(@() clifford_spherical_monogenic_gram( ...
        points, 1, alg3, struct('unitTolerance', -1)), ...
        'clifford_spherical_monogenic_gram:InvalidTolerance');
    local_assert_error(@() clifford_spherical_monogenic_gram( ...
        points, 1, alg3, struct('normalizePoints', 2)), ...
        'clifford_spherical_monogenic_gram:InvalidBooleanOption');
    badAlg = struct('m', 3, 'nBlades', 8);
    local_assert_error(@() clifford_spherical_monogenic_gram( ...
        points, 1, badAlg), ...
        'clifford_spherical_monogenic_gram:InvalidAlgebra');
    local_assert_error(@() clifford_spherical_monogenic_kernel( ...
        [1, 0], [1, 0], 1, alg3), ...
        'clifford_spherical_monogenic_kernel:InvalidPoint');
    local_assert_error(@() clifford_spherical_monogenic_kernel( ...
        [2, 0, 0], [1, 0, 0], 1, alg3), ...
        'clifford_spherical_monogenic_kernel:NonUnitPoint');
end

function local_assert_star_hermitian(entries, alg)
    pointCount = size(entries, 1);
    for row = 1:pointCount
        for column = 1:pointCount
            entry = reshape(entries(row, column, :), alg.nBlades, 1);
            reverseEntry = reshape(entries(column, row, :), alg.nBlades, 1);
            local_assert_close(clifford_multivector_star(entry, alg), reverseEntry);
        end
    end
end

function local_assert_only_scalar_and_bivector_grades(entries, alg)
    for bladeIndex = 1:alg.nBlades
        mask = alg.indexToMask(bladeIndex);
        if alg.grade(mask) ~= 0 && alg.grade(mask) ~= 2
            local_assert_close(entries(:, :, bladeIndex), ...
                zeros(size(entries, 1), size(entries, 2)));
        end
    end
end

function local_assert_same_real_values(actual, expected)
    assert(max(abs(imag(actual(:)))) <= 1e-10);
    local_assert_close(sort(real(actual(:))), sort(real(expected(:))));
end

function local_assert_close(actual, expected)
    assert(isequal(size(actual), size(expected)));
    scale = max([1; abs(actual(:)); abs(expected(:))]);
    residual = norm(actual(:) - expected(:), inf);
    dimensionFactor = max(1, sqrt(max(numel(actual), numel(expected))));
    assert(residual <= 5000 * dimensionFactor * eps(scale));
end

function local_assert_error(action, expectedIdentifier)
    didError = false;
    try
        action();
    catch exception
        didError = true;
        assert(strcmp(exception.identifier, expectedIdentifier));
    end
    assert(didError);
end
