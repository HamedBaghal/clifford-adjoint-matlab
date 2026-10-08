function results = test_clifford_matrix_left_regular_matrix(dimensions)
%TEST_CLIFFORD_MATRIX_LEFT_REGULAR_MATRIX Test block left-regular matrices.
%
%   RESULTS = TEST_CLIFFORD_MATRIX_LEFT_REGULAR_MATRIX() tests m=4 and m=5.
%   Its reference construction uses signed generator matrices directly from
%   bit masks; it does not call ALG.multiplyIndices or the map under test.

    if nargin == 0
        dimensions = [4, 5];
    end

    if ~(isnumeric(dimensions) && isreal(dimensions) && isvector(dimensions) && ...
            all(isfinite(dimensions)) && all(dimensions == floor(dimensions)) && ...
            all(dimensions >= 1))
        error('test_clifford_matrix_left_regular_matrix:InvalidDimensions', ...
            'DIMENSIONS must be a vector of positive integer dimensions.');
    end

    local_edge_and_input_checks();
    savedRandomState = rng;
    restoreRandomState = onCleanup(@() rng(savedRandomState)); %#ok<NASGU>

    rowCount = 2;
    columnCount = 3;
    productColumnCount = 2;
    template = struct('m', [], 'nBlades', [], 'complexRows', [], ...
        'complexColumns', [], 'referenceResidual', [], 'actionResidual', [], ...
        'homomorphismResidual', [], 'starAdjointResidual', [], ...
        'cellTensorResidual', [], 'scalarKronResidual', []);
    results = repmat(template, 1, numel(dimensions));

    for testNumber = 1:numel(dimensions)
        m = dimensions(testNumber);
        alg = clifford_algebra(m);
        nBlades = alg.nBlades;
        generatorMatrices = local_generator_matrices(m);
        bladeMatrices = local_blade_matrices(generatorMatrices, m);

        rng(7000 + m, 'twister');
        a = local_random_tensor(rowCount, columnCount, nBlades);
        b = local_random_tensor(columnCount, productColumnCount, nBlades);

        [leftRegularA, info] = clifford_matrix_left_regular_matrix(a, alg);
        referenceA = local_reference_block_matrix(a, bladeMatrices);
        referenceResidual = norm(leftRegularA - referenceA, inf);
        local_assert_close(leftRegularA, referenceA);
        assert(isequal(size(leftRegularA), ...
            [rowCount * nBlades, columnCount * nBlades]));
        assert(strcmp(info.representation, 'block-left-regular'));
        assert(info.m == m);
        assert(info.nBlades == nBlades);
        assert(info.blockSize == nBlades);
        assert(info.rowCount == rowCount);
        assert(info.columnCount == columnCount);
        assert(isequal(info.outputSize, ...
            [rowCount * nBlades, columnCount * nBlades]));
        assert(~info.isSquare);
        assert(strcmp(info.inputFormat, 'coefficient-tensor'));
        local_assert_close(norm(leftRegularA, 'fro')^2, ...
            nBlades * sum(abs(a(:)).^2));

        % The first column of every scalar left-regular block is the entry.
        local_first_column_checks(a, leftRegularA);

        % The cell-array convenience form has exactly the same meaning.
        cellA = local_tensor_to_cell(a);
        [leftRegularCellA, cellInfo] = ...
            clifford_matrix_left_regular_matrix(cellA, alg);
        cellTensorResidual = norm(leftRegularCellA - leftRegularA, inf);
        local_assert_close(leftRegularCellA, leftRegularA);
        assert(strcmp(cellInfo.inputFormat, 'cell'));

        % Scalar-only entries must produce the ordinary scalar matrix
        % tensored with the N-by-N identity in the stated block ordering.
        scalarMatrix = randn(rowCount, columnCount) + ...
            1i * randn(rowCount, columnCount);
        scalarEntries = complex(zeros(rowCount, columnCount, nBlades));
        scalarEntries(:, :, 1) = scalarMatrix;
        scalarBlock = clifford_matrix_left_regular_matrix(scalarEntries, alg);
        scalarKronResidual = norm(scalarBlock - ...
            kron(scalarMatrix, eye(nBlades)), inf);
        local_assert_close(scalarBlock, kron(scalarMatrix, eye(nBlades)));

        % The matrix action agrees with a direct Clifford matrix-vector product.
        x = local_random_tensor(columnCount, 1, nBlades);
        packedX = local_pack_column(x);
        actualAction = leftRegularA * packedX;
        expectedAction = local_pack_column( ...
            local_matrix_vector_product(a, x, alg));
        actionResidual = norm(actualAction - expectedAction, inf);
        local_assert_close(actualAction, expectedAction);

        % Block multiplication becomes ordinary complex matrix multiplication.
        ab = local_matrix_product(a, b, alg);
        leftRegularB = clifford_matrix_left_regular_matrix(b, alg);
        leftRegularAB = clifford_matrix_left_regular_matrix(ab, alg);
        homomorphismResidual = norm(leftRegularAB - ...
            leftRegularA * leftRegularB, inf);
        local_assert_close(leftRegularAB, leftRegularA * leftRegularB);

        % Complex linearity is preserved entry-by-entry.
        d = local_random_tensor(rowCount, columnCount, nBlades);
        alpha = 2 - 3i;
        beta = -1 + 4i;
        local_assert_close(clifford_matrix_left_regular_matrix( ...
            alpha * a + beta * d, alg), alpha * leftRegularA + ...
            beta * clifford_matrix_left_regular_matrix(d, alg));

        % The matrix Clifford adjoint includes a block transpose.
        aStar = local_matrix_star(a, alg);
        leftRegularAStar = clifford_matrix_left_regular_matrix(aStar, alg);
        starAdjointResidual = norm(leftRegularAStar - leftRegularA', inf);
        local_assert_close(leftRegularAStar, leftRegularA');
        local_assert_close(clifford_matrix_left_regular_matrix( ...
            local_matrix_product(aStar, a, alg), alg), leftRegularA' * leftRegularA);

        % The block identity maps to the ordinary identity matrix.
        identityEntries = local_identity_tensor(rowCount, nBlades);
        local_assert_close(clifford_matrix_left_regular_matrix( ...
            identityEntries, alg), eye(rowCount * nBlades));
        squareEntries = local_random_tensor(rowCount, rowCount, nBlades);
        squareBlock = clifford_matrix_left_regular_matrix(squareEntries, alg);
        local_assert_close(trace(squareBlock), nBlades * ...
            sum(diag(squareEntries(:, :, 1))));

        results(testNumber).m = m;
        results(testNumber).nBlades = nBlades;
        results(testNumber).complexRows = rowCount * nBlades;
        results(testNumber).complexColumns = columnCount * nBlades;
        results(testNumber).referenceResidual = referenceResidual;
        results(testNumber).actionResidual = actionResidual;
        results(testNumber).homomorphismResidual = homomorphismResidual;
        results(testNumber).starAdjointResidual = starAdjointResidual;
        results(testNumber).cellTensorResidual = cellTensorResidual;
        results(testNumber).scalarKronResidual = scalarKronResidual;
        fprintf(['Cl_{0,%d}: %d-by-%d Clifford matrix -> %d-by-%d ', ...
            'complex block map passed.\n'], m, rowCount, columnCount, ...
            rowCount * nBlades, columnCount * nBlades);
    end
end

function entries = local_random_tensor(rowCount, columnCount, nBlades)
    entries = randn(rowCount, columnCount, nBlades) + ...
        1i * randn(rowCount, columnCount, nBlades);
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

function local_first_column_checks(entries, blockMatrix)
    rowCount = size(entries, 1);
    columnCount = size(entries, 2);
    nBlades = size(entries, 3);
    for row = 1:rowCount
        rowIndices = (row - 1) * nBlades + (1:nBlades);
        for column = 1:columnCount
            firstColumn = (column - 1) * nBlades + 1;
            expected = reshape(entries(row, column, :), nBlades, 1);
            local_assert_close(blockMatrix(rowIndices, firstColumn), expected);
        end
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

function output = local_matrix_vector_product(matrixEntries, vectorEntries, alg)
    output = local_matrix_product(matrixEntries, vectorEntries, alg);
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

function entries = local_identity_tensor(order, nBlades)
    entries = complex(zeros(order, order, nBlades));
    for diagonal = 1:order
        entries(diagonal, diagonal, 1) = 1;
    end
end

function packed = local_pack_column(entries)
    rowCount = size(entries, 1);
    columnCount = size(entries, 2);
    nBlades = size(entries, 3);
    assert(columnCount == 1);
    packed = zeros(rowCount * nBlades, 1);
    for row = 1:rowCount
        indices = (row - 1) * nBlades + (1:nBlades);
        packed(indices) = reshape(entries(row, 1, :), nBlades, 1);
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

function reference = local_reference_block_matrix(entries, bladeMatrices)
    rowCount = size(entries, 1);
    columnCount = size(entries, 2);
    nBlades = size(entries, 3);
    reference = zeros(rowCount * nBlades, columnCount * nBlades);
    for row = 1:rowCount
        rowIndices = (row - 1) * nBlades + (1:nBlades);
        for column = 1:columnCount
            columnIndices = (column - 1) * nBlades + (1:nBlades);
            coefficientVector = reshape(entries(row, column, :), nBlades, 1);
            block = zeros(nBlades, nBlades);
            for bladeIndex = 1:nBlades
                if coefficientVector(bladeIndex) ~= 0
                    block = block + coefficientVector(bladeIndex) * ...
                        full(bladeMatrices{bladeIndex});
                end
            end
            reference(rowIndices, columnIndices) = block;
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

function local_edge_and_input_checks()
    % For m=0, the map is exactly the ordinary complex matrix itself.
    alg0 = clifford_algebra(0);
    scalarMatrix = [2 - 3i, -1 + 4i; 5, 6 + 2i];
    [actual0, info0] = clifford_matrix_left_regular_matrix(scalarMatrix, alg0);
    local_assert_close(actual0, scalarMatrix);
    assert(strcmp(info0.inputFormat, ...
        'coefficient-tensor (implicit singleton component)'));
    assert(isequal(info0.outputSize, [2, 2]));

    cellScalarMatrix = {scalarMatrix(1,1), scalarMatrix(1,2); ...
        scalarMatrix(2,1), scalarMatrix(2,2)};
    local_assert_close(clifford_matrix_left_regular_matrix( ...
        cellScalarMatrix, alg0), scalarMatrix);

    % A single e_1 block establishes the r-by-s-by-N orientation at m=1.
    alg1 = clifford_algebra(1);
    entries1 = zeros(2, 3, 2);
    entries1(1, 2, 2) = 1;
    actual1 = clifford_matrix_left_regular_matrix(entries1, alg1);
    expected1 = zeros(4, 6);
    expected1(1:2, 3:4) = [0, -1; 1, 0];
    assert(isequal(actual1, expected1));
    singleE1 = reshape([0; 1], [1, 1, 2]);
    assert(isequal(clifford_matrix_left_regular_matrix(singleE1, alg1), ...
        [0, -1; 1, 0]));

    alg2 = clifford_algebra(2);
    nBlades = alg2.nBlades;
    valid = zeros(2, 3, nBlades);
    local_assert_error(@() clifford_matrix_left_regular_matrix( ...
        randn(2, 3, nBlades - 1), alg2), ...
        'clifford_matrix_left_regular_matrix:InvalidCoefficientTensor');
    local_assert_error(@() clifford_matrix_left_regular_matrix( ...
        randn(nBlades, 2, 3), alg2), ...
        'clifford_matrix_left_regular_matrix:InvalidCoefficientTensor');
    local_assert_error(@() clifford_matrix_left_regular_matrix( ...
        randn(2, 3, nBlades, 2), alg2), ...
        'clifford_matrix_left_regular_matrix:InvalidCoefficientTensor');
    local_assert_error(@() clifford_matrix_left_regular_matrix( ...
        randn(1, nBlades), alg2), ...
        'clifford_matrix_left_regular_matrix:InvalidCoefficientTensor');

    nonFinite = valid;
    nonFinite(1, 1, 1) = NaN;
    local_assert_error(@() clifford_matrix_left_regular_matrix(nonFinite, alg2), ...
        'clifford_matrix_left_regular_matrix:NonFiniteCoefficient');
    local_assert_error(@() clifford_matrix_left_regular_matrix( ...
        zeros(0, 2, nBlades), alg2), ...
        'clifford_matrix_left_regular_matrix:InvalidCoefficientTensor');
    local_assert_error(@() clifford_matrix_left_regular_matrix('abcd', alg2), ...
        'clifford_matrix_left_regular_matrix:InvalidCoefficientTensor');
    local_assert_error(@() clifford_matrix_left_regular_matrix(valid, struct()), ...
        'clifford_matrix_left_regular_matrix:InvalidAlgebra');

    invalidCell = cell(2, 2);
    invalidCell{1,1} = zeros(nBlades, 1);
    invalidCell{1,2} = zeros(nBlades, 1);
    invalidCell{2,1} = zeros(nBlades, 1);
    invalidCell{2,2} = zeros(nBlades - 1, 1);
    local_assert_error(@() clifford_matrix_left_regular_matrix(invalidCell, alg2), ...
        'clifford_matrix_left_regular_matrix:InvalidCellEntry');

    nonFiniteCell = cell(1, 1);
    nonFiniteCell{1,1} = [NaN; zeros(nBlades - 1, 1)];
    local_assert_error(@() clifford_matrix_left_regular_matrix( ...
        nonFiniteCell, alg2), ...
        'clifford_matrix_left_regular_matrix:NonFiniteCoefficient');
    local_assert_error(@() clifford_matrix_left_regular_matrix( ...
        cell(0, 2), alg2), ...
        'clifford_matrix_left_regular_matrix:InvalidCellMatrix');
    local_assert_error(@() clifford_matrix_left_regular_matrix( ...
        cell(2, 2, 2), alg2), ...
        'clifford_matrix_left_regular_matrix:InvalidCellMatrix');
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
