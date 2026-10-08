function results = test_clifford_matrix_from_left_regular_matrix(dimensions)
%TEST_CLIFFORD_MATRIX_FROM_LEFT_REGULAR_MATRIX Test block decoding/validation.
%
%   RESULTS = TEST_CLIFFORD_MATRIX_FROM_LEFT_REGULAR_MATRIX() tests m=4 and
%   m=5.  The report-mode projection is checked with an independent
%   Hilbert-Schmidt basis oracle, not with the decoder's extraction loops.

    if nargin == 0
        dimensions = [4, 5];
    end
    if ~(isnumeric(dimensions) && isreal(dimensions) && isvector(dimensions) && ...
            all(isfinite(dimensions)) && all(dimensions == floor(dimensions)) && ...
            all(dimensions >= 2))
        error('test_clifford_matrix_from_left_regular_matrix:InvalidDimensions', ...
            'DIMENSIONS must be a vector of integer dimensions at least 2.');
    end

    savedRandomState = rng;
    restoreRandomState = onCleanup(@() rng(savedRandomState)); %#ok<NASGU>
    local_edge_and_option_checks();

    rowCount = 2;
    columnCount = 3;
    template = struct('m', [], 'nBlades', [], 'complexRows', [], ...
        'complexColumns', [], 'exactRecoveryResidual', [], ...
        'projectionResidual', [], 'firstColumnTrapResidual', []);
    results = repmat(template, 1, numel(dimensions));

    for testNumber = 1:numel(dimensions)
        m = dimensions(testNumber);
        alg = clifford_algebra(m);
        nBlades = alg.nBlades;
        complexRows = rowCount * nBlades;
        complexColumns = columnCount * nBlades;

        % Exact rectangular block map: decoding is the inverse on the image.
        rng(10000 + m, 'twister');
        entries = local_random_tensor(rowCount, columnCount, nBlades);
        blockLeftRegular = clifford_matrix_left_regular_matrix(entries, alg);
        [decoded, info] = clifford_matrix_from_left_regular_matrix( ...
            blockLeftRegular, alg);
        local_assert_close(decoded, entries);
        local_assert_close(info.reconstructedBlockLeftRegularMatrix, ...
            blockLeftRegular);
        local_assert_close(clifford_matrix_left_regular_matrix(decoded, alg), ...
            blockLeftRegular);
        local_assert_close(info.absoluteResidual, 0);
        local_assert_close(info.relativeResidual, 0);
        assert(info.isInBlockLeftRegularImage);
        assert(strcmp(info.validationMode, 'error'));
        assert(strcmp(info.projectionMethod, ...
            'blockwise Frobenius-orthogonal projection'));
        assert(info.m == m);
        assert(info.nBlades == nBlades);
        assert(info.blockSize == nBlades);
        assert(info.frobeniusScaling == sqrt(nBlades));
        assert(info.rowCount == rowCount);
        assert(info.columnCount == columnCount);
        assert(~info.isSquare);
        assert(isequal(info.inputSize, [complexRows, complexColumns]));
        assert(isequal(info.coefficientTensorSize, ...
            [rowCount, columnCount, nBlades]));
        assert(isequal(size(decoded), [rowCount, columnCount, nBlades]));
        assert(all(info.blockAbsoluteResiduals(:) <= 1e-10));
        assert(all(info.blockRelativeResiduals(:) <= 1e-10));
        local_assert_close(norm(blockLeftRegular, 'fro')^2, ...
            nBlades * sum(abs(decoded(:)).^2));

        % A one-by-one exact map agrees with the established scalar decoder.
        singleEntry = reshape(entries(1, 1, :), nBlades, 1);
        singleMap = clifford_left_regular_matrix(singleEntry, alg);
        [singleTensor, singleInfo] = ...
            clifford_matrix_from_left_regular_matrix(singleMap, alg);
        [scalarDecoded, scalarResidual] = ...
            clifford_multivector_from_left_regular_matrix(singleMap, alg);
        local_assert_close(reshape(singleTensor, nBlades, 1), scalarDecoded);
        local_assert_close(reshape(singleTensor, nBlades, 1), singleEntry);
        local_assert_close(singleInfo.absoluteResidual, 0);
        local_assert_close(scalarResidual, 0);

        % A generic nonimage matrix is projected with an independent basis
        % oracle. Strict mode rejects it; report mode diagnoses it.
        rng(11000 + m, 'twister');
        arbitrary = randn(complexRows, complexColumns) + ...
            1i * randn(complexRows, complexColumns);
        [expectedEntries, expectedProjection] = ...
            local_projection_oracle(arbitrary, alg);
        [projectedEntries, projectionInfo] = ...
            clifford_matrix_from_left_regular_matrix(arbitrary, alg, ...
            struct('validationMode', 'report'));
        local_assert_close(projectedEntries, expectedEntries);
        local_assert_close(projectionInfo.reconstructedBlockLeftRegularMatrix, ...
            expectedProjection);
        expectedAbsoluteResidual = norm(arbitrary - expectedProjection, 'fro');
        local_assert_close(projectionInfo.absoluteResidual, ...
            expectedAbsoluteResidual);
        local_assert_close(projectionInfo.relativeResidual, ...
            expectedAbsoluteResidual / max(1, norm(arbitrary, 'fro')));
        assert(~projectionInfo.isInBlockLeftRegularImage);
        local_assert_error(@() clifford_matrix_from_left_regular_matrix( ...
            arbitrary, alg), ...
            'clifford_matrix_from_left_regular_matrix:NotInBlockLeftRegularImage');

        % This perturbation changes no local first column and is orthogonal to
        % every left-regular basis matrix. It must still be rejected after a
        % full-block validation, and report projection recovers the original A.
        delta = 1e-6;
        trap = blockLeftRegular;
        targetRow = 1;
        targetColumn = 2;
        rowIndices = (targetRow - 1) * nBlades + (1:nBlades);
        columnIndices = (targetColumn - 1) * nBlades + (1:nBlades);
        trap(rowIndices(2), columnIndices(2)) = ...
            trap(rowIndices(2), columnIndices(2)) + delta;
        trap(rowIndices(3), columnIndices(3)) = ...
            trap(rowIndices(3), columnIndices(3)) - delta;
        local_assert_close(trap(rowIndices, columnIndices(1)), ...
            blockLeftRegular(rowIndices, columnIndices(1)));
        [trapEntries, trapInfo] = clifford_matrix_from_left_regular_matrix( ...
            trap, alg, struct('validationMode', 'report'));
        local_assert_close(trapEntries, entries);
        local_assert_close(trapInfo.reconstructedBlockLeftRegularMatrix, ...
            blockLeftRegular);
        local_assert_close(trapInfo.absoluteResidual, sqrt(2) * delta);
        local_assert_close(trapInfo.blockAbsoluteResiduals(targetRow, targetColumn), ...
            sqrt(2) * delta);
        expectedBlockResiduals = zeros(rowCount, columnCount);
        expectedBlockResiduals(targetRow, targetColumn) = sqrt(2) * delta;
        local_assert_close(trapInfo.blockAbsoluteResiduals, ...
            expectedBlockResiduals);
        assert(~trapInfo.isInBlockLeftRegularImage);
        local_assert_error(@() clifford_matrix_from_left_regular_matrix( ...
            trap, alg), ...
            'clifford_matrix_from_left_regular_matrix:NotInBlockLeftRegularImage');

        % Acceptance thresholds use the residual after orthogonal projection.
        residualScale = max(1, norm(trap, 'fro'));
        residual = trapInfo.absoluteResidual;
        relativeResidual = residual / residualScale;
        [~, absoluteAccept] = clifford_matrix_from_left_regular_matrix( ...
            trap, alg, struct('validationMode', 'report', ...
            'absoluteTolerance', 2 * residual, 'relativeTolerance', 0));
        assert(absoluteAccept.isInBlockLeftRegularImage);
        [~, absoluteReject] = clifford_matrix_from_left_regular_matrix( ...
            trap, alg, struct('validationMode', 'report', ...
            'absoluteTolerance', 0.25 * residual, 'relativeTolerance', 0));
        assert(~absoluteReject.isInBlockLeftRegularImage);
        clifford_matrix_from_left_regular_matrix(trap, alg, ...
            struct('absoluteTolerance', 2 * residual, 'relativeTolerance', 0));
        local_assert_error(@() clifford_matrix_from_left_regular_matrix( ...
            trap, alg, struct('absoluteTolerance', 0.25 * residual, ...
            'relativeTolerance', 0)), ...
            'clifford_matrix_from_left_regular_matrix:NotInBlockLeftRegularImage');
        clifford_matrix_from_left_regular_matrix(trap, alg, 2 * relativeResidual);
        local_assert_error(@() clifford_matrix_from_left_regular_matrix( ...
            trap, alg, 0.25 * relativeResidual), ...
            'clifford_matrix_from_left_regular_matrix:NotInBlockLeftRegularImage');
        clifford_matrix_from_left_regular_matrix(trap, alg, ...
            struct('absoluteTolerance', 0.6 * residual, ...
            'relativeTolerance', 0.6 * relativeResidual));
        local_assert_error(@() clifford_matrix_from_left_regular_matrix( ...
            trap, alg, struct('absoluteTolerance', 0.2 * residual, ...
            'relativeTolerance', 0.2 * relativeResidual)), ...
            'clifford_matrix_from_left_regular_matrix:NotInBlockLeftRegularImage');

        results(testNumber).m = m;
        results(testNumber).nBlades = nBlades;
        results(testNumber).complexRows = complexRows;
        results(testNumber).complexColumns = complexColumns;
        results(testNumber).exactRecoveryResidual = info.relativeResidual;
        results(testNumber).projectionResidual = projectionInfo.relativeResidual;
        results(testNumber).firstColumnTrapResidual = trapInfo.relativeResidual;
        fprintf(['Cl_{0,%d}: %d-by-%d complex block map; ', ...
            'decoder and image validation passed.\n'], m, complexRows, ...
            complexColumns);
    end
end

function entries = local_random_tensor(rowCount, columnCount, nBlades)
    entries = randn(rowCount, columnCount, nBlades) + ...
        1i * randn(rowCount, columnCount, nBlades);
end

function [entries, projection] = local_projection_oracle(blockMatrix, alg)
% Independent Hilbert-Schmidt projection using the tested scalar basis maps.

    nBlades = alg.nBlades;
    rowCount = size(blockMatrix, 1) / nBlades;
    columnCount = size(blockMatrix, 2) / nBlades;
    basisMatrices = cell(nBlades, 1);
    for bladeIndex = 1:nBlades
        basisVector = zeros(nBlades, 1);
        basisVector(bladeIndex) = 1;
        basisMatrices{bladeIndex} = ...
            clifford_left_regular_matrix(basisVector, alg);
    end

    entries = complex(zeros(rowCount, columnCount, nBlades));
    projection = complex(zeros(size(blockMatrix)));
    for row = 1:rowCount
        rowIndices = (row - 1) * nBlades + (1:nBlades);
        for column = 1:columnCount
            columnIndices = (column - 1) * nBlades + (1:nBlades);
            block = blockMatrix(rowIndices, columnIndices);
            coefficientVector = complex(zeros(nBlades, 1));
            projectedBlock = zeros(nBlades, nBlades);
            for bladeIndex = 1:nBlades
                basisMatrix = basisMatrices{bladeIndex};
                coefficientVector(bladeIndex) = ...
                    sum(conj(basisMatrix(:)) .* block(:)) / nBlades;
                projectedBlock = projectedBlock + ...
                    coefficientVector(bladeIndex) * basisMatrix;
            end
            entries(row, column, :) = reshape( ...
                coefficientVector, [1, 1, nBlades]);
            projection(rowIndices, columnIndices) = projectedBlock;
        end
    end
end

function local_edge_and_option_checks()
    % At m=0 every ordinary complex matrix is already a block image.
    alg0 = clifford_algebra(0);
    ordinary = [2 - 3i, -1 + 4i, 7; 5, 6 + 2i, -2i];
    [ordinaryEntries, ordinaryInfo] = ...
        clifford_matrix_from_left_regular_matrix(ordinary, alg0);
    local_assert_close(ordinaryEntries, ordinary);
    local_assert_close(ordinaryInfo.reconstructedBlockLeftRegularMatrix, ordinary);
    local_assert_close(ordinaryInfo.absoluteResidual, 0);
    assert(ordinaryInfo.isInBlockLeftRegularImage);
    assert(size(ordinaryEntries, 3) == 1);

    % At m=1, perturbing one non-first block entry demonstrates the true
    % orthogonal projection, rather than first-column extraction.
    alg1 = clifford_algebra(1);
    entries1 = zeros(2, 3, 2);
    entries1(1, 2, 2) = 1;
    block1 = clifford_matrix_left_regular_matrix(entries1, alg1);
    local_assert_close(block1(1:2, 3:4), [0, -1; 1, 0]);
    [decoded1, info1] = clifford_matrix_from_left_regular_matrix(block1, alg1);
    local_assert_close(decoded1, entries1);
    local_assert_close(info1.absoluteResidual, 0);

    delta = 1e-6;
    noisy1 = block1;
    noisy1(2, 4) = noisy1(2, 4) + delta;
    local_assert_close(noisy1(1:2, 3), block1(1:2, 3));
    [reportEntries1, reportInfo1] = ...
        clifford_matrix_from_left_regular_matrix(noisy1, alg1, ...
        struct('validationMode', 'report', 'relativeTolerance', 0));
    expected1 = entries1;
    expected1(1, 2, 1) = delta / 2;
    local_assert_close(reportEntries1, expected1);
    local_assert_close(reportInfo1.reconstructedBlockLeftRegularMatrix(1:2, 3:4), ...
        [delta / 2, -1; 1, delta / 2]);
    local_assert_close(reportInfo1.absoluteResidual, delta / sqrt(2));
    assert(~reportInfo1.isInBlockLeftRegularImage);
    local_assert_error(@() clifford_matrix_from_left_regular_matrix( ...
        noisy1, alg1, 0), ...
        'clifford_matrix_from_left_regular_matrix:NotInBlockLeftRegularImage');

    alg2 = clifford_algebra(2);
    nBlades = alg2.nBlades;
    local_assert_error(@() clifford_matrix_from_left_regular_matrix( ...
        zeros(7, 8), alg2), ...
        'clifford_matrix_from_left_regular_matrix:IncompatibleDimensions');
    local_assert_error(@() clifford_matrix_from_left_regular_matrix( ...
        zeros(8, 7), alg2), ...
        'clifford_matrix_from_left_regular_matrix:IncompatibleDimensions');
    local_assert_error(@() clifford_matrix_from_left_regular_matrix( ...
        zeros(0, 4), alg2), ...
        'clifford_matrix_from_left_regular_matrix:InvalidBlockMatrix');
    local_assert_error(@() clifford_matrix_from_left_regular_matrix( ...
        zeros(4, 4, 2), alg2), ...
        'clifford_matrix_from_left_regular_matrix:InvalidBlockMatrix');
    local_assert_error(@() clifford_matrix_from_left_regular_matrix( ...
        'abcd', alg2), ...
        'clifford_matrix_from_left_regular_matrix:InvalidBlockMatrix');
    local_assert_error(@() clifford_matrix_from_left_regular_matrix( ...
        cell(4, 4), alg2), ...
        'clifford_matrix_from_left_regular_matrix:InvalidBlockMatrix');
    nonFinite = zeros(nBlades, nBlades);
    nonFinite(1, 1) = NaN;
    local_assert_error(@() clifford_matrix_from_left_regular_matrix( ...
        nonFinite, alg2), ...
        'clifford_matrix_from_left_regular_matrix:NonFiniteEntry');
    local_assert_error(@() clifford_matrix_from_left_regular_matrix( ...
        zeros(nBlades, nBlades), struct()), ...
        'clifford_matrix_from_left_regular_matrix:InvalidAlgebra');

    valid = eye(nBlades);
    local_assert_error(@() clifford_matrix_from_left_regular_matrix( ...
        valid, alg2, [1, 2]), ...
        'clifford_matrix_from_left_regular_matrix:InvalidOptions');
    local_assert_error(@() clifford_matrix_from_left_regular_matrix( ...
        valid, alg2, -1), ...
        'clifford_matrix_from_left_regular_matrix:InvalidTolerance');
    local_assert_error(@() clifford_matrix_from_left_regular_matrix( ...
        valid, alg2, 1i), ...
        'clifford_matrix_from_left_regular_matrix:InvalidTolerance');
    local_assert_error(@() clifford_matrix_from_left_regular_matrix( ...
        valid, alg2, NaN), ...
        'clifford_matrix_from_left_regular_matrix:InvalidTolerance');
    local_assert_error(@() clifford_matrix_from_left_regular_matrix( ...
        valid, alg2, struct('unknown', true)), ...
        'clifford_matrix_from_left_regular_matrix:UnknownOption');
    local_assert_error(@() clifford_matrix_from_left_regular_matrix( ...
        valid, alg2, struct('relativeTolerance', -1)), ...
        'clifford_matrix_from_left_regular_matrix:InvalidTolerance');
    local_assert_error(@() clifford_matrix_from_left_regular_matrix( ...
        valid, alg2, struct('absoluteTolerance', 1i)), ...
        'clifford_matrix_from_left_regular_matrix:InvalidTolerance');
    local_assert_error(@() clifford_matrix_from_left_regular_matrix( ...
        valid, alg2, struct('validationMode', 'invalid')), ...
        'clifford_matrix_from_left_regular_matrix:InvalidValidationMode');
    local_assert_error(@() clifford_matrix_from_left_regular_matrix( ...
        valid, alg2, struct('validationMode', char('error', 'report'))), ...
        'clifford_matrix_from_left_regular_matrix:InvalidValidationMode');
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
