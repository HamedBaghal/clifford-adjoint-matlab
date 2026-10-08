function results = test_clifford_matrix_left_regular_eigensystem(dimensions)
%TEST_CLIFFORD_MATRIX_LEFT_REGULAR_EIGENSYSTEM Test Clifford-matrix spectra.
%
%   RESULTS = TEST_CLIFFORD_MATRIX_LEFT_REGULAR_EIGENSYSTEM() tests m=4 and
%   m=5.  The main reference block matrix is assembled by direct packed
%   Clifford matrix-vector actions; it does not call the block-map routine.

    if nargin == 0
        dimensions = [4, 5];
    end
    if ~(isnumeric(dimensions) && isreal(dimensions) && isvector(dimensions) && ...
            all(isfinite(dimensions)) && all(dimensions == floor(dimensions)) && ...
            all(dimensions >= 2))
        error('test_clifford_matrix_left_regular_eigensystem:InvalidDimensions', ...
            'DIMENSIONS must be a vector of integer dimensions at least 2.');
    end

    savedRandomState = rng;
    restoreRandomState = onCleanup(@() rng(savedRandomState)); %#ok<NASGU>
    local_edge_and_option_checks();

    rowCount = 2;
    template = struct('m', [], 'nBlades', [], 'complexDimension', [], ...
        'maxRelativeEigenResidual', [], 'sigmaMin', [], ...
        'conditionNumber2', [], 'positiveLambdaMin', [], ...
        'positiveConditionNumber2', [], 'nonnormalConditionNumber2', []);
    results = repmat(template, 1, numel(dimensions));

    for testNumber = 1:numel(dimensions)
        m = dimensions(testNumber);
        alg = clifford_algebra(m);
        nBlades = alg.nBlades;
        complexDimension = rowCount * nBlades;

        % Generic dense data tests the raw eigensystem and the r-by-N-by-q
        % unpacking convention for every returned packed eigenvector.
        rng(8000 + m, 'twister');
        entries = local_random_tensor(rowCount, rowCount, nBlades);
        info = clifford_matrix_left_regular_eigensystem(entries, alg);
        reference = local_reference_block_matrix(entries, alg);
        local_assert_close(info.blockLeftRegularMatrix, reference);
        local_assert_close(info.leftRegularMatrix, reference);
        local_assert_close(info.coefficientTensor, entries);
        local_assert_close(info.matrixStarCoefficientTensor, ...
            local_matrix_star(entries, alg));
        assert(strcmp(info.representation, 'block-left-regular'));
        assert(info.m == m);
        assert(info.nBlades == nBlades);
        assert(info.cliffordMatrixSize == rowCount);
        assert(info.complexDimension == complexDimension);
        assert(strcmp(info.inputFormat, 'coefficient-tensor'));
        assert(isequal(size(info.rightEigenvectors), ...
            [complexDimension, complexDimension]));
        assert(isequal(size(info.rightEigenvectorCoefficients), ...
            [rowCount, nBlades, complexDimension]));
        local_assert_same_complex_values(info.eigenvalues, eig(reference));
        local_assert_close(info.singularValues, svd(reference));
        local_assert_close(sum(info.eigenvalues), trace(reference));
        local_assert_eigenpairs(entries, alg, info, reference);
        assert(info.maxRelativeEigenResidual <= 1e-8);
        assert(abs(info.maxRelativeEigenResidual - ...
            max(info.relativeEigenResiduals)) <= 1e-14);
        assert(info.numericalRank == complexDimension);
        assert(~info.isNumericallySingular);
        assert(info.isNumericallyInvertible);
        assert(isfinite(info.rawConditionNumber2));
        assert(isfinite(info.conditionNumber2));
        local_assert_close(info.conditionNumber2, info.sigmaMax / info.sigmaMin);
        local_assert_regular_representation(info, m, rowCount);

        % The cell convenience input has exactly the same Clifford matrix.
        cellEntries = local_tensor_to_cell(entries);
        cellInfo = clifford_matrix_left_regular_eigensystem(cellEntries, alg);
        local_assert_close(cellInfo.blockLeftRegularMatrix, reference);
        local_assert_same_complex_values(cellInfo.eigenvalues, info.eigenvalues);
        local_assert_close(cellInfo.singularValues, info.singularValues);
        assert(strcmp(cellInfo.inputFormat, 'cell'));

        % A one-by-one matrix is exactly the existing scalar left action.
        oneByOne = reshape(entries(1, 1, :), [1, 1, nBlades]);
        oneByOneInfo = clifford_matrix_left_regular_eigensystem(oneByOne, alg);
        singleEntry = reshape(entries(1, 1, :), nBlades, 1);
        local_assert_close(oneByOneInfo.blockLeftRegularMatrix, ...
            clifford_left_regular_matrix(singleEntry, alg));
        local_assert_same_complex_values(oneByOneInfo.eigenvalues, ...
            eig(clifford_left_regular_matrix(singleEntry, alg)));

        % U is an invertible but nonnormal scalar-valued Clifford matrix.
        % All its eigenvalues are one, yet sigmaMin exposes solve sensitivity.
        unipotentEntries = local_scalar_matrix_tensor([1, 2; 0, 1], nBlades);
        unipotentInfo = clifford_matrix_left_regular_eigensystem( ...
            unipotentEntries, alg);
        assert(~unipotentInfo.isNumericallySingular);
        assert(~unipotentInfo.isNumericallyNormal);
        assert(isfinite(unipotentInfo.conditionNumber2));
        assert(unipotentInfo.conditionNumber2 > 1.1);
        assert(max(abs(unipotentInfo.eigenvalues - 1)) < 1e-7);
        assert(unipotentInfo.sigmaMin < 0.9);
        assert(abs(min(abs(unipotentInfo.eigenvalues)) - ...
            unipotentInfo.sigmaMin) > 0.05);

        % A singular scalar-valued matrix has rank N rather than r*N.
        singularEntries = local_scalar_matrix_tensor([1, 0; 0, 0], nBlades);
        singularInfo = clifford_matrix_left_regular_eigensystem( ...
            singularEntries, alg);
        assert(singularInfo.numericalRank == nBlades);
        assert(singularInfo.isNumericallySingular);
        assert(~singularInfo.isNumericallyInvertible);
        assert(isinf(singularInfo.conditionNumber2));
        assert(singularInfo.sigmaMin <= singularInfo.singularValueTolerance);

        % H=B^star*B+2I is self-star and positive definite.  Its Hermitian
        % branch is the correct source of lambdaMin and the PD condition number.
        rng(9000 + m, 'twister');
        b = local_random_tensor(rowCount, rowCount, nBlades);
        bStar = local_matrix_star(b, alg);
        positiveEntries = local_matrix_product(bStar, b, alg);
        positiveEntries = local_add_scalar_identity(positiveEntries, 2);
        positiveInfo = clifford_matrix_left_regular_eigensystem( ...
            positiveEntries, alg);
        referenceB = local_reference_block_matrix(b, alg);
        local_assert_close(positiveInfo.blockLeftRegularMatrix, ...
            referenceB' * referenceB + 2 * eye(complexDimension));
        local_assert_positive_hermitian_analysis( ...
            positiveInfo, positiveEntries, alg, 2);

        % This exact scalar diagonal matrix isolates the forced N-fold copies.
        diagonalEntries = local_scalar_matrix_tensor([2, 0; 0, 5], nBlades);
        diagonalInfo = clifford_matrix_left_regular_eigensystem( ...
            diagonalEntries, alg);
        local_assert_diagonal_hermitian_analysis(diagonalInfo, nBlades);

        results(testNumber).m = m;
        results(testNumber).nBlades = nBlades;
        results(testNumber).complexDimension = complexDimension;
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
        fprintf(['Cl_{0,%d}: %d-by-%d Clifford-matrix eigensystem; ', ...
            'packed spectral diagnostics passed.\n'], m, complexDimension, ...
            complexDimension);
    end
end

function entries = local_random_tensor(rowCount, columnCount, nBlades)
    entries = randn(rowCount, columnCount, nBlades) + ...
        1i * randn(rowCount, columnCount, nBlades);
end

function entries = local_scalar_matrix_tensor(matrix, nBlades)
    entries = complex(zeros(size(matrix, 1), size(matrix, 2), nBlades));
    entries(:, :, 1) = matrix;
end

function cellEntries = local_tensor_to_cell(entries)
    rowCount = size(entries, 1);
    columnCount = size(entries, 2);
    nBlades = size(entries, 3);
    cellEntries = cell(rowCount, columnCount);
    for row = 1:rowCount
        for column = 1:columnCount
            cellEntries{row, column} = reshape( ...
                entries(row, column, :), nBlades, 1);
        end
    end
end

function local_assert_eigenpairs(entries, alg, info, reference)
    rowCount = size(entries, 1);
    nBlades = alg.nBlades;
    complexDimension = rowCount * nBlades;
    assert(isequal(size(info.eigenvalues), [complexDimension, 1]));
    assert(numel(info.relativeEigenResiduals) == complexDimension);

    for valueNumber = 1:complexDimension
        value = info.eigenvalues(valueNumber);
        packedVector = info.packedRightEigenvectors(:, valueNumber);
        assert(abs(norm(packedVector) - 1) <= 1e-10);
        local_assert_close(reference * packedVector, value * packedVector);

        vectorEntries = local_unpack_packed_column( ...
            packedVector, rowCount, nBlades);
        local_assert_close(local_pack_column(vectorEntries), packedVector);
        coefficientSlice = info.rightEigenvectorCoefficients(:, :, valueNumber);
        local_assert_close(coefficientSlice, ...
            reshape(packedVector, nBlades, rowCount).');
        local_assert_close(local_pack_coefficient_slice(coefficientSlice), ...
            packedVector);
        local_assert_close(local_pack_column(local_matrix_vector_product( ...
            entries, vectorEntries, alg)), value * packedVector);

        denominator = (info.sigmaMax + abs(value)) * norm(packedVector);
        residual = norm(reference * packedVector - value * packedVector);
        if denominator == 0
            expectedRelativeResidual = residual;
        else
            expectedRelativeResidual = residual / denominator;
        end
        local_assert_close(info.relativeEigenResiduals(valueNumber), ...
            expectedRelativeResidual);
    end
end

function local_assert_regular_representation(info, m, rowCount)
    d = 2^floor(m / 2);
    metadata = info.regularRepresentation;
    assert(metadata.cliffordIrreducibleBlockDimension == d);
    assert(metadata.matrixIrreducibleBlockDimension == rowCount * d);
    assert(metadata.regularCopyMultiplicity == d);
    assert(metadata.fullDimension == info.complexDimension);
    if mod(m, 2) == 0
        assert(~metadata.isOddDimension);
        assert(metadata.minimalFaithfulDimension == rowCount * d);
    else
        assert(metadata.isOddDimension);
        assert(metadata.minimalFaithfulDimension == 2 * rowCount * d);
    end
end

function local_assert_positive_hermitian_analysis(info, entries, alg, lowerBound)
    complexDimension = info.complexDimension;
    hermitian = info.hermitian;
    assert(info.isNumericallyStarSelfAdjoint);
    assert(info.isNumericallyNormal);
    assert(hermitian.isAvailable);
    assert(hermitian.isNumericallyPositiveDefinite);
    assert(hermitian.lambdaMin >= lowerBound - 1e-9);
    assert(hermitian.lambdaMax >= hermitian.lambdaMin);
    local_assert_close(hermitian.symmetrizedCoefficientTensor, entries);
    local_assert_close(hermitian.symmetrizedBlockLeftRegularMatrix, ...
        info.blockLeftRegularMatrix);
    local_assert_close(hermitian.symmetrizedBlockLeftRegularMatrix, ...
        clifford_matrix_left_regular_matrix( ...
        hermitian.symmetrizedCoefficientTensor, alg));
    local_assert_close(hermitian.rightEigenvectors' * ...
        hermitian.rightEigenvectors, eye(complexDimension));
    local_assert_close(hermitian.numericallyHermitianMatrix * ...
        hermitian.rightEigenvectors, hermitian.rightEigenvectors * ...
        diag(hermitian.eigenvalues));
    local_assert_close(hermitian.numericallyHermitianMatrix, ...
        hermitian.rightEigenvectors * diag(hermitian.eigenvalues) * ...
        hermitian.rightEigenvectors');
    local_assert_close(info.conditionNumber2, ...
        hermitian.positiveDefiniteConditionNumber2);
    assert(hermitian.maxRelativeEigenResidual <= 1e-8);
    assert(hermitian.unitarityResidual <= 1e-8);
    assert(hermitian.spectralReconstructionResidual <= 1e-8);

    for valueNumber = 1:complexDimension
        coefficientSlice = ...
            hermitian.rightEigenvectorCoefficients(:, :, valueNumber);
        local_assert_close(local_pack_coefficient_slice(coefficientSlice), ...
            hermitian.rightEigenvectors(:, valueNumber));
    end
    assert(isequal(size(hermitian.symmetrizedCoefficientTensor), ...
        size(entries)));
    assert(hermitian.symmetrizedMapHermitianResidual <= 1e-12);
end

function local_assert_diagonal_hermitian_analysis(info, nBlades)
    hermitian = info.hermitian;
    assert(info.isNumericallyStarSelfAdjoint);
    assert(info.isNumericallyNormal);
    assert(hermitian.isAvailable);
    assert(hermitian.isNumericallyPositiveDefinite);
    local_assert_close(hermitian.lambdaMin, 2);
    local_assert_close(hermitian.lambdaMax, 5);
    local_assert_close(hermitian.positiveDefiniteConditionNumber2, 5 / 2);
    local_assert_close(info.conditionNumber2, 5 / 2);
    local_assert_close(hermitian.eigenvalues, ...
        [2 * ones(nBlades, 1); 5 * ones(nBlades, 1)]);
    clusters = hermitian.eigenvalueClusters;
    assert(numel(clusters) == 2);
    local_assert_close(clusters(1).centroid, 2);
    local_assert_close(clusters(2).centroid, 5);
    assert(clusters(1).multiplicity == nBlades);
    assert(clusters(2).multiplicity == nBlades);
end

function reference = local_reference_block_matrix(entries, alg)
% Build every packed column from direct Clifford matrix-vector products.

    rowCount = size(entries, 1);
    columnCount = size(entries, 2);
    nBlades = alg.nBlades;
    reference = complex(zeros(rowCount * nBlades, columnCount * nBlades));
    for inputRow = 1:columnCount
        for bladeIndex = 1:nBlades
            basisVector = complex(zeros(columnCount, 1, nBlades));
            basisVector(inputRow, 1, bladeIndex) = 1;
            output = local_matrix_vector_product(entries, basisVector, alg);
            outputColumn = (inputRow - 1) * nBlades + bladeIndex;
            reference(:, outputColumn) = local_pack_column(output);
        end
    end
end

function output = local_matrix_vector_product(matrixEntries, vectorEntries, alg)
    rowCount = size(matrixEntries, 1);
    columnCount = size(matrixEntries, 2);
    nBlades = alg.nBlades;
    assert(size(vectorEntries, 1) == columnCount);
    assert(size(vectorEntries, 2) == 1);
    output = complex(zeros(rowCount, 1, nBlades));
    for row = 1:rowCount
        entry = zeros(nBlades, 1);
        for column = 1:columnCount
            left = reshape(matrixEntries(row, column, :), nBlades, 1);
            right = reshape(vectorEntries(column, 1, :), nBlades, 1);
            entry = entry + clifford_multivector_product(left, right, alg);
        end
        output(row, 1, :) = reshape(entry, [1, 1, nBlades]);
    end
end

function product = local_matrix_product(left, right, alg)
    leftRows = size(left, 1);
    innerDimension = size(left, 2);
    rightColumns = size(right, 2);
    nBlades = alg.nBlades;
    assert(size(right, 1) == innerDimension);
    product = complex(zeros(leftRows, rightColumns, nBlades));
    for row = 1:leftRows
        for column = 1:rightColumns
            entry = zeros(nBlades, 1);
            for inner = 1:innerDimension
                leftEntry = reshape(left(row, inner, :), nBlades, 1);
                rightEntry = reshape(right(inner, column, :), nBlades, 1);
                entry = entry + clifford_multivector_product( ...
                    leftEntry, rightEntry, alg);
            end
            product(row, column, :) = reshape(entry, [1, 1, nBlades]);
        end
    end
end

function starEntries = local_matrix_star(entries, alg)
    rowCount = size(entries, 1);
    columnCount = size(entries, 2);
    nBlades = alg.nBlades;
    starEntries = complex(zeros(columnCount, rowCount, nBlades));
    for row = 1:rowCount
        for column = 1:columnCount
            entry = reshape(entries(row, column, :), nBlades, 1);
            starEntries(column, row, :) = reshape( ...
                clifford_multivector_star(entry, alg), [1, 1, nBlades]);
        end
    end
end

function entries = local_add_scalar_identity(entries, scalar)
    rowCount = size(entries, 1);
    assert(size(entries, 2) == rowCount);
    for diagonal = 1:rowCount
        entries(diagonal, diagonal, 1) = entries(diagonal, diagonal, 1) + scalar;
    end
end

function packed = local_pack_column(entries)
    rowCount = size(entries, 1);
    columnCount = size(entries, 2);
    nBlades = size(entries, 3);
    assert(columnCount == 1);
    packed = complex(zeros(rowCount * nBlades, 1));
    for row = 1:rowCount
        indices = (row - 1) * nBlades + (1:nBlades);
        packed(indices) = reshape(entries(row, 1, :), nBlades, 1);
    end
end

function entries = local_unpack_packed_column(packed, rowCount, nBlades)
    entries = complex(zeros(rowCount, 1, nBlades));
    for row = 1:rowCount
        indices = (row - 1) * nBlades + (1:nBlades);
        entries(row, 1, :) = reshape(packed(indices), [1, 1, nBlades]);
    end
end

function packed = local_pack_coefficient_slice(coefficients)
    rowCount = size(coefficients, 1);
    nBlades = size(coefficients, 2);
    packed = complex(zeros(rowCount * nBlades, 1));
    for row = 1:rowCount
        indices = (row - 1) * nBlades + (1:nBlades);
        packed(indices) = coefficients(row, :).';
    end
end

function local_edge_and_option_checks()
    % At m=0, this is exactly an ordinary complex matrix eigensystem.
    alg0 = clifford_algebra(0);
    ordinary = [2, 1 - 2i; 0, -3];
    ordinaryInfo = clifford_matrix_left_regular_eigensystem(ordinary, alg0);
    local_assert_close(ordinaryInfo.blockLeftRegularMatrix, ordinary);
    local_assert_same_complex_values(ordinaryInfo.eigenvalues, eig(ordinary));
    assert(ordinaryInfo.cliffordMatrixSize == 2);
    assert(ordinaryInfo.complexDimension == 2);
    assert(ordinaryInfo.numericalRank == 2);

    % At m=1, a one-by-one e_1 matrix has the expected two central values.
    alg1 = clifford_algebra(1);
    e1 = reshape([0; 1], [1, 1, 2]);
    e1Info = clifford_matrix_left_regular_eigensystem(e1, alg1);
    local_assert_same_complex_values(e1Info.eigenvalues, [-1i; 1i]);
    local_assert_close(e1Info.blockLeftRegularMatrix, [0, -1; 1, 0]);
    assert(~e1Info.isNumericallyStarSelfAdjoint);

    alg2 = clifford_algebra(2);
    nBlades = alg2.nBlades;
    zeroEntries = zeros(2, 2, nBlades);
    zeroInfo = clifford_matrix_left_regular_eigensystem(zeroEntries, alg2);
    assert(zeroInfo.numericalRank == 0);
    assert(zeroInfo.isNumericallySingular);
    assert(isinf(zeroInfo.conditionNumber2));
    assert(zeroInfo.isNumericallyNormal);
    assert(zeroInfo.isNumericallyStarSelfAdjoint);
    assert(zeroInfo.hermitian.isAvailable);
    assert(~zeroInfo.hermitian.isNumericallyPositiveDefinite);
    local_assert_close(zeroInfo.hermitian.lambdaMin, 0);

    tinyDiagonal = local_scalar_matrix_tensor([1, 0; 0, 1e-8], nBlades);
    tinyInfo = clifford_matrix_left_regular_eigensystem(tinyDiagonal, alg2, ...
        struct('singularValueTolerance', 1e-6));
    assert(tinyInfo.numericalRank == nBlades);
    assert(tinyInfo.isNumericallySingular);
    assert(isinf(tinyInfo.conditionNumber2));

    local_assert_error(@() clifford_matrix_left_regular_eigensystem( ...
        zeros(2, 3, nBlades), alg2), ...
        'clifford_matrix_left_regular_eigensystem:NonSquareInput');
    local_assert_error(@() clifford_matrix_left_regular_eigensystem( ...
        zeros(2, 2, nBlades), alg2, 1), ...
        'clifford_matrix_left_regular_eigensystem:InvalidOptions');
    local_assert_error(@() clifford_matrix_left_regular_eigensystem( ...
        zeros(2, 2, nBlades), alg2, struct('notAnOption', true)), ...
        'clifford_matrix_left_regular_eigensystem:UnknownOption');
    local_assert_error(@() clifford_matrix_left_regular_eigensystem( ...
        zeros(2, 2, nBlades), alg2, struct('relativeTolerance', -1)), ...
        'clifford_matrix_left_regular_eigensystem:InvalidTolerance');
    local_assert_error(@() clifford_matrix_left_regular_eigensystem( ...
        zeros(2, 2, nBlades), alg2, ...
        struct('normalityTolerance', 1i)), ...
        'clifford_matrix_left_regular_eigensystem:InvalidTolerance');
    local_assert_error(@() clifford_matrix_left_regular_eigensystem( ...
        zeros(2, 2, nBlades), alg2, ...
        struct('absoluteTolerance', NaN)), ...
        'clifford_matrix_left_regular_eigensystem:InvalidTolerance');
    local_assert_error(@() clifford_matrix_left_regular_eigensystem( ...
        zeros(2, 2, nBlades), struct()), ...
        'clifford_matrix_left_regular_matrix:InvalidAlgebra');
    local_assert_error(@() clifford_matrix_left_regular_eigensystem( ...
        zeros(2, 2, nBlades - 1), alg2), ...
        'clifford_matrix_left_regular_matrix:InvalidCoefficientTensor');
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
