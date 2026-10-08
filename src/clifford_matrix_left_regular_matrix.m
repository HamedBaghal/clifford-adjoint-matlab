function [blockLeftRegular, info] = clifford_matrix_left_regular_matrix(entries, alg)
%CLIFFORD_MATRIX_LEFT_REGULAR_MATRIX Block left-regular map of a Clifford matrix.
%
%   BLOCKLEFTREGULAR = CLIFFORD_MATRIX_LEFT_REGULAR_MATRIX(ENTRIES, ALG)
%   maps an r-by-s matrix A=[a_ij] with entries in the complexification of
%   Cl_{0,m} to the ordinary complex block matrix
%
%       Lambda(A) = [ lambda_m(a_ij) ]_(i,j),
%
%   where lambda_m is CLIFFORD_LEFT_REGULAR_MATRIX.  If N=ALG.nBlades, the
%   result has size (r*N)-by-(s*N).  Its block (i,j) is N-by-N and represents
%   left multiplication by a_ij in the ordered blade-coefficient basis.
%
%   The canonical numeric input is an r-by-s-by-N coefficient tensor:
%
%       ENTRIES(i,j,k) is the coefficient of blade mask k-1 in a_ij.
%
%   For m=0 (N=1), an ordinary r-by-s numeric matrix is accepted with the
%   singleton coefficient dimension implicit.  As a convenience, ENTRIES may
%   instead be an r-by-s cell array whose cell {i,j} is a finite numeric
%   coefficient vector of length N.  The two input forms produce the same
%   matrix.  The orientation r-by-s-by-N is deliberate: the first two indices
%   are the Clifford-matrix row and column, and the last index is the blade.
%
%   The output acts on a packed s-tuple X=(x_1,...,x_s)^T of multivectors:
%
%       [coeff(y_1); ...; coeff(y_r)] = Lambda(A) *
%                                        [coeff(x_1); ...; coeff(x_s)],
%
%   where y_i=sum_j a_ij*x_j.  For conformable Clifford matrices A and B,
%
%       Lambda(A*B) = Lambda(A)*Lambda(B).
%
%   For a square A, define (A^star)_ij=(a_ji)^star.  Then
%
%       Lambda(A^star) = Lambda(A)'.
%
%   The map is faithful, but it is not onto all complex matrices: every
%   N-by-N block must belong to the scalar left-regular image.  It is the
%   canonical full reference representation, not a future compact tau_m map.
%
%   [BLOCKLEFTREGULAR, INFO] also returns dimensions, input format, and the
%   coefficient/block ordering convention.

    local_validate_algebra(alg);
    nBlades = alg.nBlades;
    [coefficientTensor, rowCount, columnCount, inputFormat] = ...
        local_normalize_entries(entries, nBlades);

    blockLeftRegular = zeros(rowCount * nBlades, columnCount * nBlades);
    for row = 1:rowCount
        rowIndices = (row - 1) * nBlades + (1:nBlades);
        for column = 1:columnCount
            columnIndices = (column - 1) * nBlades + (1:nBlades);
            coefficientVector = reshape( ...
                coefficientTensor(row, column, :), nBlades, 1);
            blockLeftRegular(rowIndices, columnIndices) = ...
                clifford_left_regular_matrix(coefficientVector, alg);
        end
    end

    info = struct();
    info.representation = 'block-left-regular';
    info.m = alg.m;
    info.nBlades = nBlades;
    info.blockSize = nBlades;
    info.rowCount = rowCount;
    info.columnCount = columnCount;
    info.outputSize = [rowCount * nBlades, columnCount * nBlades];
    info.isSquare = rowCount == columnCount;
    info.inputFormat = inputFormat;
    info.coefficientConvention = ...
        'entries(i,j,k) is blade k coefficient of Clifford entry (i,j).';
    info.vectorizationConvention = ...
        '[coeff(x_1); ...; coeff(x_s)] with blade coefficients in mask order.';
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
        error('clifford_matrix_left_regular_matrix:InvalidAlgebra', ...
            ['ALG must be a structure returned by clifford_algebra, with ', ...
             'a valid multiplyIndices function handle.']);
    end
end

function [coefficientTensor, rowCount, columnCount, inputFormat] = ...
        local_normalize_entries(entries, nBlades)

    if iscell(entries)
        [coefficientTensor, rowCount, columnCount] = ...
            local_normalize_cell_entries(entries, nBlades);
        inputFormat = 'cell';
        return;
    end

    if ~(isnumeric(entries) && ~isempty(entries))
        error('clifford_matrix_left_regular_matrix:InvalidCoefficientTensor', ...
            ['ENTRIES must be a nonempty finite numeric r-by-s-by-N tensor ', ...
             'or an r-by-s cell array of coefficient vectors.']);
    end
    if ~all(isfinite(entries(:)))
        error('clifford_matrix_left_regular_matrix:NonFiniteCoefficient', ...
            'ENTRIES contains a NaN or Inf coefficient.');
    end

    if nBlades == 1 && ismatrix(entries)
        rowCount = size(entries, 1);
        columnCount = size(entries, 2);
        coefficientTensor = reshape(double(entries), ...
            [rowCount, columnCount, 1]);
        inputFormat = 'coefficient-tensor (implicit singleton component)';
        return;
    end

    hasTensorShape = ndims(entries) == 3 && size(entries, 3) == nBlades;
    if ~hasTensorShape || size(entries, 1) < 1 || size(entries, 2) < 1
        error('clifford_matrix_left_regular_matrix:InvalidCoefficientTensor', ...
            ['For N=ALG.nBlades, numeric ENTRIES must have size r-by-s-by-N ', ...
             'with r and s positive.']);
    end

    rowCount = size(entries, 1);
    columnCount = size(entries, 2);
    coefficientTensor = double(entries);
    inputFormat = 'coefficient-tensor';
end

function [coefficientTensor, rowCount, columnCount] = ...
        local_normalize_cell_entries(entries, nBlades)

    if ~(ismatrix(entries) && ~isempty(entries))
        error('clifford_matrix_left_regular_matrix:InvalidCellMatrix', ...
            'Cell ENTRIES must be a nonempty two-dimensional cell array.');
    end

    rowCount = size(entries, 1);
    columnCount = size(entries, 2);
    coefficientTensor = complex(zeros(rowCount, columnCount, nBlades));

    for row = 1:rowCount
        for column = 1:columnCount
            value = entries{row, column};
            if ~(isnumeric(value) && isvector(value) && ...
                    numel(value) == nBlades)
                error('clifford_matrix_left_regular_matrix:InvalidCellEntry', ...
                    ['Each ENTRIES{i,j} must be a numeric coefficient vector ', ...
                     'with exactly ALG.nBlades entries.']);
            end
            if ~all(isfinite(value(:)))
                error('clifford_matrix_left_regular_matrix:NonFiniteCoefficient', ...
                    'ENTRIES{%d,%d} contains a NaN or Inf coefficient.', ...
                    row, column);
            end
            coefficientTensor(row, column, :) = reshape( ...
                double(value(:)), [1, 1, nBlades]);
        end
    end
end
