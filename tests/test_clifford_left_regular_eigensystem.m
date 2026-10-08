function results = test_clifford_left_regular_eigensystem(dimensions)
%TEST_CLIFFORD_LEFT_REGULAR_EIGENSYSTEM Test regular-representation spectra.
%
%   RESULTS = TEST_CLIFFORD_LEFT_REGULAR_EIGENSYSTEM() tests m=4 and m=5.
%   It also tests the m=0 and m=1 edge cases, a nonnormal invertible element,
%   a nonzero zero divisor, and the self-star positive-definite branch.

    if nargin == 0
        dimensions = [4, 5];
    end

    if ~(isnumeric(dimensions) && isreal(dimensions) && isvector(dimensions) && ...
            all(isfinite(dimensions)) && all(dimensions == floor(dimensions)) && ...
            all(dimensions >= 2))
        error('test_clifford_left_regular_eigensystem:InvalidDimensions', ...
            'DIMENSIONS must be a vector of integer dimensions at least 2.');
    end

    local_edge_and_option_checks();
    savedRandomState = rng;
    restoreRandomState = onCleanup(@() rng(savedRandomState)); %#ok<NASGU>

    template = struct('m', [], 'nBlades', [], ...
        'maxRelativeEigenResidual', [], 'sigmaMin', [], ...
        'conditionNumber2', [], 'positiveLambdaMin', [], ...
        'positiveConditionNumber2', [], 'nonnormalConditionNumber2', []);
    results = repmat(template, 1, numel(dimensions));

    for testNumber = 1:numel(dimensions)
        m = dimensions(testNumber);
        alg = clifford_algebra(m);
        nBlades = alg.nBlades;
        one = zeros(nBlades, 1);
        one(1) = 1;

        % A generic complex multivector: raw eigenpairs are checked both as
        % matrix equations and as Clifford-product equations.
        rng(5000 + m, 'twister');
        a = randn(nBlades, 1) + 1i * randn(nBlades, 1);
        info = clifford_left_regular_eigensystem(a, alg);
        leftRegular = clifford_left_regular_matrix(a, alg);
        local_assert_close(info.leftRegularMatrix, leftRegular);
        local_assert_eigenpairs(a, alg, info);
        local_assert_close(info.singularValues, svd(leftRegular));
        local_assert_close(sum(info.eigenvalues), trace(leftRegular));
        assert(~info.isNumericallySingular);
        assert(isfinite(info.rawConditionNumber2));
        assert(isfinite(info.conditionNumber2));
        local_assert_close(info.conditionNumber2, ...
            info.sigmaMax / info.sigmaMin);
        assert(info.numericalRank == nBlades);
        assert(info.regularRepresentation.irreducibleBlockDimension == ...
            2^floor(m / 2));
        assert(info.regularRepresentation.regularCopyMultiplicity == ...
            2^floor(m / 2));
        if mod(m, 2) == 0
            assert(info.regularRepresentation.minimalFaithfulDimension == ...
                2^floor(m / 2));
        else
            assert(info.regularRepresentation.minimalFaithfulDimension == ...
                2^(floor(m / 2) + 1));
        end

        % n = e_1 + i e_2 is nilpotent.  Thus 1+n is invertible but
        % nonnormal; all of its eigenvalues are 1 while sigmaMin is smaller.
        e1 = zeros(nBlades, 1);
        e2 = zeros(nBlades, 1);
        e1(2) = 1;
        e2(3) = 1;
        nilpotent = e1 + 1i * e2;
        local_assert_close(clifford_multivector_product( ...
            nilpotent, nilpotent, alg), zeros(nBlades, 1));
        unipotent = one + nilpotent;
        unipotentInverse = one - nilpotent;
        local_assert_close(clifford_multivector_product( ...
            unipotent, unipotentInverse, alg), one);
        unipotentInfo = clifford_left_regular_eigensystem(unipotent, alg);
        assert(~unipotentInfo.isNumericallySingular);
        assert(~unipotentInfo.isNumericallyNormal);
        assert(isfinite(unipotentInfo.conditionNumber2));
        assert(unipotentInfo.conditionNumber2 > 1.1);
        assert(max(abs(unipotentInfo.eigenvalues - 1)) < 1e-6);
        assert(unipotentInfo.sigmaMin < 0.9);
        assert(abs(min(abs(unipotentInfo.eigenvalues)) - ...
            unipotentInfo.sigmaMin) > 0.05);

        % b^star*b + 2 is guaranteed self-star and positive definite because
        % its left-regular matrix is L_b^H*L_b + 2I.
        rng(6000 + m, 'twister');
        b = randn(nBlades, 1) + 1i * randn(nBlades, 1);
        bStar = clifford_multivector_star(b, alg);
        positive = clifford_multivector_product(bStar, b, alg);
        positive(1) = positive(1) + 2;
        options = struct('computeSpectralProjectors', true, ...
            'computeInverseSquareRoot', true);
        positiveInfo = clifford_left_regular_eigensystem(positive, alg, options);
        local_assert_positive_hermitian_analysis(positiveInfo, alg, 2);

        % A scalar has one exact numerical cluster of full regular
        % multiplicity.  This isolates the clustering/projector contract.
        scalarInfo = clifford_left_regular_eigensystem(3 * one, alg, ...
            struct('computeSpectralProjectors', true));
        scalarClusters = scalarInfo.hermitian.eigenvalueClusters;
        assert(numel(scalarClusters) == 1);
        assert(scalarClusters(1).multiplicity == nBlades);
        assert(isequal(sort(scalarClusters(1).indices(:)), (1:nBlades).'));
        local_assert_close(scalarClusters(1).centroid, 3);
        scalarProjector = scalarInfo.hermitian.spectralProjectors{1};
        local_assert_close(scalarProjector, eye(nBlades));
        assert(scalarInfo.hermitian.spectralProjectorIsInLeftRegularImage(1));
        local_assert_close(scalarInfo.hermitian.spectralProjectorCoefficientVectors{1}, ...
            one);

        results(testNumber).m = m;
        results(testNumber).nBlades = nBlades;
        results(testNumber).maxRelativeEigenResidual = ...
            info.maxRelativeEigenResidual;
        results(testNumber).sigmaMin = info.sigmaMin;
        results(testNumber).conditionNumber2 = info.conditionNumber2;
        results(testNumber).positiveLambdaMin = ...
            positiveInfo.hermitian.lambdaMin;
        results(testNumber).positiveConditionNumber2 = ...
            positiveInfo.hermitian.positiveDefiniteConditionNumber2;
        results(testNumber).nonnormalConditionNumber2 = ...
            unipotentInfo.conditionNumber2;
        fprintf(['Cl_{0,%d}: %d-by-%d eigensystem; ', ...
            'spectral diagnostics passed.\n'], m, nBlades, nBlades);
    end
end

function local_assert_eigenpairs(a, alg, info)
    leftRegular = info.leftRegularMatrix;
    nBlades = size(leftRegular, 1);
    assert(isequal(size(info.eigenvalues), [nBlades, 1]));
    assert(isequal(size(info.rightEigenvectors), [nBlades, nBlades]));
    assert(numel(info.relativeEigenResiduals) == nBlades);

    for valueNumber = 1:nBlades
        value = info.eigenvalues(valueNumber);
        vector = info.rightEigenvectors(:, valueNumber);
        assert(abs(norm(vector) - 1) <= 1e-10);
        local_assert_close(leftRegular * vector, value * vector);
        local_assert_close(clifford_multivector_product(a, vector, alg), ...
            value * vector);
    end
    assert(info.maxRelativeEigenResidual <= 1e-8);
    assert(abs(info.maxRelativeEigenResidual - ...
        max(info.relativeEigenResiduals)) <= 1e-14);
end

function local_assert_positive_hermitian_analysis(info, alg, lowerBound)
    nBlades = alg.nBlades;
    hermitian = info.hermitian;
    assert(info.isNumericallyStarSelfAdjoint);
    assert(info.isNumericallyNormal);
    assert(hermitian.isAvailable);
    assert(hermitian.isNumericallyPositiveDefinite);
    assert(hermitian.lambdaMin >= lowerBound - 1e-9);
    assert(hermitian.lambdaMax >= hermitian.lambdaMin);
    assert(hermitian.inverseSquareRootAvailable);
    assert(hermitian.inverseSquareRootIsInLeftRegularImage);
    assert(hermitian.projectorsAvailable);

    matrix = hermitian.symmetrizedMatrix;
    symmetrizedCoefficientVector = hermitian.symmetrizedCoefficientVector;
    vectors = hermitian.rightEigenvectors;
    values = hermitian.eigenvalues;
    local_assert_close(vectors' * vectors, eye(nBlades));
    local_assert_close(matrix * vectors, vectors * diag(values));
    local_assert_close(matrix, vectors * diag(values) * vectors');
    local_assert_close(info.conditionNumber2, ...
        hermitian.positiveDefiniteConditionNumber2);

    projectors = hermitian.spectralProjectors;
    clusters = hermitian.eigenvalueClusters;
    assert(numel(projectors) == numel(clusters));
    covered = [];
    projectorSum = zeros(nBlades, nBlades);
    for clusterNumber = 1:numel(projectors)
        projector = projectors{clusterNumber};
        indices = clusters(clusterNumber).indices(:);
        local_assert_close(projector, projector');
        local_assert_close(projector * projector, projector);
        local_assert_close(matrix * projector, projector * matrix);
        local_assert_close(trace(projector), numel(indices));
        assert(hermitian.spectralProjectorIsInLeftRegularImage( ...
            clusterNumber));
        assert(hermitian.spectralProjectorImageResiduals(clusterNumber) <= ...
            hermitian.leftRegularImageTolerance);
        projectorCoefficientVector = ...
            hermitian.spectralProjectorCoefficientVectors{clusterNumber};
        local_assert_close(clifford_left_regular_matrix( ...
            projectorCoefficientVector, alg), projector);
        local_assert_close(clifford_multivector_product( ...
            projectorCoefficientVector, projectorCoefficientVector, alg), ...
            projectorCoefficientVector);
        local_assert_close(clifford_multivector_star( ...
            projectorCoefficientVector, alg), projectorCoefficientVector);
        projectorSum = projectorSum + projector;
        covered = [covered; indices]; %#ok<AGROW>
    end
    assert(isequal(sort(covered), (1:nBlades).'));
    local_assert_close(projectorSum, eye(nBlades));
    for leftNumber = 1:numel(projectors)
        for rightNumber = leftNumber + 1:numel(projectors)
            local_assert_close(projectors{leftNumber} * projectors{rightNumber}, ...
                zeros(nBlades, nBlades));
        end
    end

    inverseSquareRoot = hermitian.inverseSquareRoot;
    local_assert_close(inverseSquareRoot, inverseSquareRoot');
    local_assert_close(inverseSquareRoot * matrix * inverseSquareRoot, ...
        eye(nBlades));
    local_assert_close(matrix * inverseSquareRoot * inverseSquareRoot, ...
        eye(nBlades));
    assert(~isempty(hermitian.inverseSquareRootCoefficientVector));
    assert(hermitian.inverseSquareRootImageResidual <= ...
        hermitian.leftRegularImageTolerance);
    local_assert_close(clifford_left_regular_matrix( ...
        hermitian.inverseSquareRootCoefficientVector, alg), inverseSquareRoot);
    local_assert_close(clifford_multivector_star( ...
        hermitian.inverseSquareRootCoefficientVector, alg), ...
        hermitian.inverseSquareRootCoefficientVector);
    local_assert_close(clifford_multivector_product( ...
        clifford_multivector_product( ...
        hermitian.inverseSquareRootCoefficientVector, ...
        symmetrizedCoefficientVector, alg), ...
        hermitian.inverseSquareRootCoefficientVector, alg), ...
        [1; zeros(nBlades - 1, 1)]);
end

function local_edge_and_option_checks()
    alg0 = clifford_algebra(0);
    scalar = 2 - 3i;
    scalarInfo = clifford_left_regular_eigensystem(scalar, alg0);
    assert(isequal(scalarInfo.leftRegularMatrix, scalar));
    local_assert_close(scalarInfo.eigenvalues, scalar);
    local_assert_close(scalarInfo.singularValues, abs(scalar));
    local_assert_close(scalarInfo.conditionNumber2, 1);
    assert(~scalarInfo.isNumericallySingular);
    assert(scalarInfo.isNumericallyNormal);
    assert(~scalarInfo.isNumericallyStarSelfAdjoint);
    assert(~scalarInfo.hermitian.isAvailable);

    alg1 = clifford_algebra(1);
    one = [1; 0];
    e1 = [0; 1];

    e1Info = clifford_left_regular_eigensystem(e1, alg1);
    assert(e1Info.isNumericallyNormal);
    assert(~e1Info.isNumericallyStarSelfAdjoint);
    local_assert_same_complex_values(e1Info.eigenvalues, [-1i; 1i]);
    local_assert_eigenpairs(e1, alg1, e1Info);

    indefinite = 1i * e1;
    indefiniteInfo = clifford_left_regular_eigensystem(indefinite, alg1, ...
        struct('computeSpectralProjectors', true, ...
        'computeInverseSquareRoot', true));
    assert(indefiniteInfo.isNumericallyStarSelfAdjoint);
    assert(indefiniteInfo.hermitian.isAvailable);
    assert(~indefiniteInfo.hermitian.isNumericallyPositiveDefinite);
    assert(~indefiniteInfo.hermitian.inverseSquareRootAvailable);
    assert(isempty(indefiniteInfo.hermitian.inverseSquareRoot));
    local_assert_close(indefiniteInfo.hermitian.eigenvalues, [-1; 1]);

    % (1+i e_1)(1-i e_1)=0: a nonzero multivector can be singular.
    zeroDivisor = one + 1i * e1;
    zeroDivisorPartner = one - 1i * e1;
    local_assert_close(clifford_multivector_product( ...
        zeroDivisor, zeroDivisorPartner, alg1), zeros(2, 1));
    zeroDivisorInfo = clifford_left_regular_eigensystem(zeroDivisor, alg1);
    assert(zeroDivisorInfo.isNumericallySingular);
    assert(~zeroDivisorInfo.isNumericallyInvertible);
    assert(isinf(zeroDivisorInfo.conditionNumber2));
    assert(zeroDivisorInfo.sigmaMin <= ...
        zeroDivisorInfo.singularValueTolerance);
    local_assert_same_complex_values(zeroDivisorInfo.eigenvalues, [0; 2]);

    alg2 = clifford_algebra(2);
    a2 = [1; 0; 0; 0];
    local_assert_error(@() clifford_left_regular_eigensystem(a2, alg2, 1), ...
        'clifford_left_regular_eigensystem:InvalidOptions');
    local_assert_error(@() clifford_left_regular_eigensystem(a2, alg2, ...
        struct('notAnOption', true)), ...
        'clifford_left_regular_eigensystem:UnknownOption');
    local_assert_error(@() clifford_left_regular_eigensystem(a2, alg2, ...
        struct('relativeTolerance', -1)), ...
        'clifford_left_regular_eigensystem:InvalidTolerance');
    local_assert_error(@() clifford_left_regular_eigensystem(a2, alg2, ...
        struct('eigenvalueClusterTolerance', 1i)), ...
        'clifford_left_regular_eigensystem:InvalidTolerance');
    local_assert_error(@() clifford_left_regular_eigensystem(a2, alg2, ...
        struct('computeSpectralProjectors', 2)), ...
        'clifford_left_regular_eigensystem:InvalidLogicalOption');
    local_assert_error(@() clifford_left_regular_eigensystem(a2, alg2, ...
        struct('computeInverseSquareRoot', [true false])), ...
        'clifford_left_regular_eigensystem:InvalidLogicalOption');
    local_assert_error(@() clifford_left_regular_eigensystem([1; 0], alg2), ...
        'clifford_left_regular_matrix:InvalidCoefficientVector');
end

function local_assert_same_complex_values(actual, expected)
    actualPairs = sortrows([real(actual(:)), imag(actual(:))], [1, 2]);
    expectedPairs = sortrows([real(expected(:)), imag(expected(:))], [1, 2]);
    local_assert_close(actualPairs, expectedPairs);
end

function local_assert_close(actual, expected)
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
