function [entries, info] = clifford_matrix_from_left_regular_matrix( ...
        blockLeftRegular, alg, options)
%CLIFFORD_MATRIX_FROM_LEFT_REGULAR_MATRIX Decode and validate a block map.
%
%   [ENTRIES, INFO] = CLIFFORD_MATRIX_FROM_LEFT_REGULAR_MATRIX(BLOCKLEFTREGULAR,
%   ALG) decodes a finite complex matrix of size (r*N)-by-(s*N), where
%   N=ALG.nBlades, against the block left-regular image
%
%       Lambda(A) = [ lambda_m(a_ij) ]_(i,j).
%
%   ENTRIES is the canonical r-by-s-by-N coefficient tensor.  Thus
%   ENTRIES(i,j,k) is the coefficient of blade mask k-1 in a_ij.  For m=0,
%   MATLAB naturally displays the trailing singleton coefficient dimension as
%   an ordinary r-by-s matrix.
%
%   The decoder uses the blockwise Frobenius-orthogonal projection onto the
%   left-regular image.  If E_p=lambda_m(e_(p-1)), then the coefficient of
%   blade p-1 in the decoded entry of one N-by-N block B is
%
%       a_p = trace(E_p' * B)/N.
%
%   Consequently, the reconstructed matrix INFO.reconstructedBlockLeftRegularMatrix
%   is the closest block left-regular matrix to BLOCKLEFTREGULAR in Frobenius
%   norm.  On an exact block map, this agrees with the simpler first-column
%   recovery because lambda_m(a)(:,1)=a.  For an arbitrary noisy matrix it
%   deliberately uses every column rather than silently trusting first columns.
%
%   By default the function validates the reconstruction and errors if the
%   input is not within the requested tolerance of the image.  To inspect an
%   out-of-image input and its closest structured reconstruction, use
%
%       options = struct('validationMode', 'report');
%       [entries, info] = clifford_matrix_from_left_regular_matrix(B, alg, options);
%
%   The INFO fields include absoluteResidual, relativeResidual,
%   acceptanceThreshold, isInBlockLeftRegularImage, and residuals for every
%   N-by-N block.
%
%   OPTIONS is optional.  A nonnegative numeric scalar is accepted as a
%   shorthand for OPTIONS.relativeTolerance with strict validation.  Otherwise
%   OPTIONS must be a scalar structure with these fields:
%
%       relativeTolerance  nonnegative dimensionless tolerance; default
%                          100*eps*max(1, max(size(BLOCKLEFTREGULAR))).
%       absoluteTolerance  nonnegative absolute Frobenius tolerance; default 0.
%       validationMode     'error' (default) or 'report'.
%
%   The acceptance test is
%
%       absoluteResidual <= absoluteTolerance + relativeTolerance *
%                           max(1, norm(BLOCKLEFTREGULAR, 'fro')).
%
%   This function is the inverse of CLIFFORD_MATRIX_LEFT_REGULAR_MATRIX only
%   on its structured image.  For an arbitrary matrix, ENTRIES represents its
%   closest structured approximation; INFO must be checked before interpreting
%   that approximation as an exact Clifford-valued matrix.

    if nargin < 3
        options = struct();
    end
    options = local_validate_options(options);
    local_validate_algebra(alg);
    nBlades = alg.nBlades;
    blockLeftRegular = local_validate_block_matrix(blockLeftRegular, nBlades);

    rowCount = size(blockLeftRegular, 1) / nBlades;
    columnCount = size(blockLeftRegular, 2) / nBlades;
    if isempty(options.relativeTolerance)
        relativeTolerance = 100 * eps * max( ...
            [1, size(blockLeftRegular, 1), size(blockLeftRegular, 2)]);
    else
        relativeTolerance = options.relativeTolerance;
    end
    absoluteTolerance = options.absoluteTolerance;

    entries = complex(zeros(rowCount, columnCount, nBlades));
    for row = 1:rowCount
        rowIndices = (row - 1) * nBlades + (1:nBlades);
        for column = 1:columnCount
            columnIndices = (column - 1) * nBlades + (1:nBlades);
            block = blockLeftRegular(rowIndices, columnIndices);
            entries(row, column, :) = reshape( ...
                local_project_block(block, alg), [1, 1, nBlades]);
        end
    end

    reconstructed = clifford_matrix_left_regular_matrix(entries, alg);
    difference = blockLeftRegular - reconstructed;
    inputFrobeniusNorm = norm(blockLeftRegular, 'fro');
    absoluteResidual = norm(difference, 'fro');
    relativeResidual = absoluteResidual / max(1, inputFrobeniusNorm);
    acceptanceThreshold = absoluteTolerance + relativeTolerance * ...
        max(1, inputFrobeniusNorm);

    [blockAbsoluteResiduals, blockRelativeResiduals] = ...
        local_block_residuals(blockLeftRegular, reconstructed, ...
        rowCount, columnCount, nBlades);

    info = struct();
    info.representation = 'block-left-regular-decoder';
    info.projectionMethod = 'blockwise Frobenius-orthogonal projection';
    info.m = alg.m;
    info.nBlades = nBlades;
    info.blockSize = nBlades;
    info.rowCount = rowCount;
    info.columnCount = columnCount;
    info.isSquare = rowCount == columnCount;
    info.inputSize = size(blockLeftRegular);
    info.coefficientTensorSize = [rowCount, columnCount, nBlades];
    info.coefficientConvention = ...
        'entries(i,j,k) is blade-mask k-1 coefficient of Clifford entry (i,j).';
    info.validationMode = options.validationMode;
    info.inputFrobeniusNorm = inputFrobeniusNorm;
    info.reconstructedFrobeniusNorm = norm(reconstructed, 'fro');
    info.reconstructedBlockLeftRegularMatrix = reconstructed;
    info.absoluteResidual = absoluteResidual;
    info.relativeResidual = relativeResidual;
    info.absoluteTolerance = absoluteTolerance;
    info.relativeTolerance = relativeTolerance;
    info.acceptanceThreshold = acceptanceThreshold;
    info.isInBlockLeftRegularImage = absoluteResidual <= acceptanceThreshold;
    info.blockAbsoluteResiduals = blockAbsoluteResiduals;
    info.blockRelativeResiduals = blockRelativeResiduals;
    info.frobeniusScaling = sqrt(nBlades);

    if ~info.isInBlockLeftRegularImage && ...
            strcmp(options.validationMode, 'error')
        error('clifford_matrix_from_left_regular_matrix:NotInBlockLeftRegularImage', ...
            ['BLOCKLEFTREGULAR is not within the requested tolerance of the ', ...
             'block left-regular Clifford image. Use validationMode=''report'' ', ...
             'to inspect its closest structured reconstruction.']);
    end
end

function options = local_validate_options(options)
    if isempty(options)
        options = struct();
    end

    if isnumeric(options) && isscalar(options)
        options = struct('relativeTolerance', options);
    end
    if ~(isstruct(options) && isscalar(options))
        error('clifford_matrix_from_left_regular_matrix:InvalidOptions', ...
            ['OPTIONS must be a scalar structure, or one finite nonnegative ', ...
             'real scalar relative tolerance.']);
    end

    allowedFields = {'relativeTolerance', 'absoluteTolerance', ...
        'validationMode'};
    suppliedFields = fieldnames(options);
    for fieldNumber = 1:numel(suppliedFields)
        if ~any(strcmp(suppliedFields{fieldNumber}, allowedFields))
            error('clifford_matrix_from_left_regular_matrix:UnknownOption', ...
                'OPTIONS contains the unsupported field "%s".', ...
                suppliedFields{fieldNumber});
        end
    end

    normalized = struct();
    normalized.relativeTolerance = [];
    normalized.absoluteTolerance = 0;
    normalized.validationMode = 'error';
    for fieldNumber = 1:numel(suppliedFields)
        fieldName = suppliedFields{fieldNumber};
        normalized.(fieldName) = options.(fieldName);
    end

    normalized.relativeTolerance = local_validate_optional_tolerance( ...
        normalized.relativeTolerance, 'relativeTolerance');
    normalized.absoluteTolerance = local_validate_optional_tolerance( ...
        normalized.absoluteTolerance, 'absoluteTolerance');
    if isempty(normalized.absoluteTolerance)
        normalized.absoluteTolerance = 0;
    end
    normalized.validationMode = local_validate_validation_mode( ...
        normalized.validationMode);
    options = normalized;
end

function value = local_validate_optional_tolerance(value, fieldName)
    if isempty(value)
        return;
    end
    if ~(isnumeric(value) && isreal(value) && isscalar(value) && ...
            isfinite(value) && value >= 0)
        error('clifford_matrix_from_left_regular_matrix:InvalidTolerance', ...
            ['OPTIONS.%s must be empty or one finite nonnegative ', ...
             'real scalar.'], fieldName);
    end
    value = double(value);
end

function validationMode = local_validate_validation_mode(validationMode)
    if isstring(validationMode) && isscalar(validationMode)
        validationMode = char(validationMode);
    end
    if ~(ischar(validationMode) && isrow(validationMode))
        error('clifford_matrix_from_left_regular_matrix:InvalidValidationMode', ...
            'OPTIONS.validationMode must be ''error'' or ''report''.');
    end
    validationMode = lower(validationMode);
    if ~(strcmp(validationMode, 'error') || strcmp(validationMode, 'report'))
        error('clifford_matrix_from_left_regular_matrix:InvalidValidationMode', ...
            'OPTIONS.validationMode must be ''error'' or ''report''.');
    end
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
        error('clifford_matrix_from_left_regular_matrix:InvalidAlgebra', ...
            ['ALG must be a structure returned by clifford_algebra, with ', ...
             'a valid multiplyIndices function handle.']);
    end
end

function blockLeftRegular = local_validate_block_matrix( ...
        blockLeftRegular, nBlades)
    isValidMatrix = isnumeric(blockLeftRegular) && ...
        ismatrix(blockLeftRegular) && ~isempty(blockLeftRegular);
    if ~isValidMatrix
        error('clifford_matrix_from_left_regular_matrix:InvalidBlockMatrix', ...
            ['BLOCKLEFTREGULAR must be a nonempty finite numeric ', ...
             'two-dimensional matrix.']);
    end
    if ~all(isfinite(blockLeftRegular(:)))
        error('clifford_matrix_from_left_regular_matrix:NonFiniteEntry', ...
            'BLOCKLEFTREGULAR contains a NaN or Inf entry.');
    end
    if mod(size(blockLeftRegular, 1), nBlades) ~= 0 || ...
            mod(size(blockLeftRegular, 2), nBlades) ~= 0
        error('clifford_matrix_from_left_regular_matrix:IncompatibleDimensions', ...
            ['Both dimensions of BLOCKLEFTREGULAR must be multiples of ', ...
             'ALG.nBlades.']);
    end
    blockLeftRegular = full(double(blockLeftRegular));
end

function coefficients = local_project_block(block, alg)
% Orthogonal projection onto span{lambda_m(e_0),...,lambda_m(e_(N-1))}.

    nBlades = alg.nBlades;
    coefficients = complex(zeros(nBlades, 1));
    for leftIndex = 1:nBlades
        innerProduct = 0;
        for rightIndex = 1:nBlades
            [signValue, outputIndex] = ...
                alg.multiplyIndices(leftIndex, rightIndex);
            innerProduct = innerProduct + ...
                signValue * block(outputIndex, rightIndex);
        end
        coefficients(leftIndex) = innerProduct / nBlades;
    end
end

function [absoluteResiduals, relativeResiduals] = local_block_residuals( ...
        original, reconstructed, rowCount, columnCount, nBlades)
    absoluteResiduals = zeros(rowCount, columnCount);
    relativeResiduals = zeros(rowCount, columnCount);
    for row = 1:rowCount
        rowIndices = (row - 1) * nBlades + (1:nBlades);
        for column = 1:columnCount
            columnIndices = (column - 1) * nBlades + (1:nBlades);
            originalBlock = original(rowIndices, columnIndices);
            difference = originalBlock - reconstructed(rowIndices, columnIndices);
            absoluteResiduals(row, column) = norm(difference, 'fro');
            relativeResiduals(row, column) = absoluteResiduals(row, column) / ...
                max(1, norm(originalBlock, 'fro'));
        end
    end
end
