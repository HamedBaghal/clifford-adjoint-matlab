function spectrum = clifford_matrix_left_regular_eigensystem(entries, alg, options)
%CLIFFORD_MATRIX_LEFT_REGULAR_EIGENSYSTEM Spectrum of a Clifford matrix.
%
%   SPECTRUM = CLIFFORD_MATRIX_LEFT_REGULAR_EIGENSYSTEM(ENTRIES, ALG)
%   accepts a square matrix A=[a_ij] with entries in the complexification of
%   Cl_{0,m}, in either input form accepted by
%   CLIFFORD_MATRIX_LEFT_REGULAR_MATRIX.  It constructs the faithful block
%   left-regular matrix
%
%       Lambda(A) = [ lambda_m(a_ij) ]_(i,j)
%
%   and computes its ordinary complex eigenvalues and right eigenvectors.
%   If V=SPECTRUM.packedRightEigenvectors and zeta=SPECTRUM.eigenvalues, then
%   column j of V packs an r-tuple X_j=(x_1,...,x_r)^T of multivectors with
%
%       A*X_j = zeta(j)*X_j.
%
%   The eigenvalue zeta(j) is an ordinary central complex scalar.  This is a
%   utility for the full packed left-action problem.  It does not assert a
%   theory of noncentral or right-Clifford eigenvalues.  The raw multiplicity
%   and individual eigenvectors depend on this block left-regular
%   representation.
%
%   For z in C,
%
%       z is in the central complex spectrum of A
%       <=> z*I-A is not invertible in M_r(C tensor Cl_{0,m})
%       <=> z*I-Lambda(A) is singular.
%
%   The returned SPECTRUM.rightEigenvectorCoefficients has size
%   r-by-N-by-(r*N), where N=ALG.nBlades.  Its entry (i,k,j) is the
%   coefficient of blade mask k-1 in component x_i of the j-th packed
%   eigenvector.  Equivalently, for one j use
%
%       coefficients = reshape(SPECTRUM.packedRightEigenvectors(:,j), N, r).'
%
%   SPECTRUM = CLIFFORD_MATRIX_LEFT_REGULAR_EIGENSYSTEM(ENTRIES, ALG, OPTIONS)
%   accepts a scalar structure with these optional fields:
%
%       relativeTolerance           relative numerical resolution; default
%                                   100*(r*N)*eps.
%       absoluteTolerance           absolute numerical resolution; default 0.
%       eigenvalueClusterTolerance  maximum diameter of a greedy numerical
%                                   cluster; default ABS + REL*sigmaMax.
%       singularValueTolerance      numerical rank cutoff; same default.
%       positiveDefiniteTolerance   cutoff for lambdaMin on the Hermitian
%                                   branch; same default.
%       selfAdjointTolerance        relative star-self-adjoint test; default
%                                   REL.
%       normalityTolerance          relative normality test; default REL.
%
%   Tolerances are diagnostics; they never modify the raw input or suppress
%   raw eigenvalues.  For a nonnormal matrix use sigmaMin and conditionNumber2,
%   rather than min(abs(eigenvalues)), to assess inversion stability.

    if nargin < 3
        options = struct();
    end
    options = local_validate_options(options);

    [blockLeftRegularMatrix, mapInfo] = ...
        clifford_matrix_left_regular_matrix(entries, alg);
    if ~mapInfo.isSquare
        error('clifford_matrix_left_regular_eigensystem:NonSquareInput', ...
            ['ENTRIES must represent a square Clifford-valued matrix for an ', ...
             'eigensystem calculation.']);
    end

    rowCount = mapInfo.rowCount;
    nBlades = mapInfo.nBlades;
    complexDimension = rowCount * nBlades;
    coefficientTensor = local_decode_coefficient_tensor( ...
        blockLeftRegularMatrix, rowCount, nBlades);
    matrixStarCoefficientTensor = local_matrix_star( ...
        coefficientTensor, alg);

    singularValues = svd(blockLeftRegularMatrix);
    sigmaMax = singularValues(1);
    sigmaMin = singularValues(end);

    if isempty(options.relativeTolerance)
        relativeTolerance = 100 * complexDimension * eps;
    else
        relativeTolerance = options.relativeTolerance;
    end
    absoluteTolerance = options.absoluteTolerance;
    defaultValueTolerance = absoluteTolerance + relativeTolerance * sigmaMax;
    eigenvalueClusterTolerance = local_choose_tolerance( ...
        options.eigenvalueClusterTolerance, defaultValueTolerance);
    singularValueTolerance = local_choose_tolerance( ...
        options.singularValueTolerance, defaultValueTolerance);
    positiveDefiniteTolerance = local_choose_tolerance( ...
        options.positiveDefiniteTolerance, defaultValueTolerance);
    selfAdjointTolerance = local_choose_tolerance( ...
        options.selfAdjointTolerance, relativeTolerance);
    normalityTolerance = local_choose_tolerance( ...
        options.normalityTolerance, relativeTolerance);

    [packedRightEigenvectors, eigenvalueMatrix] = eig(blockLeftRegularMatrix);
    eigenvalues = diag(eigenvalueMatrix);
    packedRightEigenvectors = local_normalize_columns(packedRightEigenvectors);
    relativeEigenResiduals = local_eigen_residuals(blockLeftRegularMatrix, ...
        eigenvalues, packedRightEigenvectors, sigmaMax);

    numericalRank = sum(singularValues > singularValueTolerance);
    isNumericallySingular = numericalRank < complexDimension;
    if sigmaMin == 0 || sigmaMax == 0
        rawConditionNumber2 = Inf;
    else
        rawConditionNumber2 = sigmaMax / sigmaMin;
    end
    if isNumericallySingular
        conditionNumber2 = Inf;
    else
        conditionNumber2 = rawConditionNumber2;
    end

    starSelfAdjointResidual = local_relative_tensor_difference( ...
        coefficientTensor, matrixStarCoefficientTensor);
    hermitianMatrixResidual = local_relative_matrix_difference( ...
        blockLeftRegularMatrix, blockLeftRegularMatrix');
    normalityCommutator = blockLeftRegularMatrix' * blockLeftRegularMatrix - ...
        blockLeftRegularMatrix * blockLeftRegularMatrix';
    normalityResidual = local_relative_matrix_norm(normalityCommutator, ...
        norm(blockLeftRegularMatrix, 'fro')^2);
    isNumericallyStarSelfAdjoint = ...
        starSelfAdjointResidual <= selfAdjointTolerance && ...
        hermitianMatrixResidual <= selfAdjointTolerance;
    isNumericallyNormal = normalityResidual <= normalityTolerance;

    spectrum = struct();
    spectrum.representation = 'block-left-regular';
    spectrum.m = alg.m;
    spectrum.nBlades = nBlades;
    spectrum.cliffordMatrixSize = rowCount;
    spectrum.complexDimension = complexDimension;
    spectrum.inputFormat = mapInfo.inputFormat;
    spectrum.mapInfo = mapInfo;
    spectrum.coefficientTensor = coefficientTensor;
    spectrum.matrixStarCoefficientTensor = matrixStarCoefficientTensor;
    spectrum.blockLeftRegularMatrix = blockLeftRegularMatrix;
    spectrum.leftRegularMatrix = blockLeftRegularMatrix;
    spectrum.eigenvalues = eigenvalues;
    spectrum.packedRightEigenvectors = packedRightEigenvectors;
    spectrum.rightEigenvectors = packedRightEigenvectors;
    spectrum.rightEigenvectorCoefficients = local_unpack_right_eigenvectors( ...
        packedRightEigenvectors, rowCount, nBlades);
    spectrum.relativeEigenResiduals = relativeEigenResiduals;
    spectrum.maxRelativeEigenResidual = max(relativeEigenResiduals);
    spectrum.eigenvectorMatrixConditionNumber2 = cond(packedRightEigenvectors);
    spectrum.eigenvalueClusters = local_cluster_values( ...
        eigenvalues, eigenvalueClusterTolerance);
    spectrum.eigenvalueClusterTolerance = eigenvalueClusterTolerance;
    spectrum.singularValues = singularValues;
    spectrum.sigmaMax = sigmaMax;
    spectrum.sigmaMin = sigmaMin;
    spectrum.singularValueTolerance = singularValueTolerance;
    spectrum.numericalRank = numericalRank;
    spectrum.isNumericallySingular = isNumericallySingular;
    spectrum.isNumericallyInvertible = ~isNumericallySingular;
    spectrum.rawConditionNumber2 = rawConditionNumber2;
    spectrum.conditionNumber2 = conditionNumber2;
    spectrum.relativeTolerance = relativeTolerance;
    spectrum.absoluteTolerance = absoluteTolerance;
    spectrum.starSelfAdjointResidual = starSelfAdjointResidual;
    spectrum.hermitianMatrixResidual = hermitianMatrixResidual;
    spectrum.selfAdjointTolerance = selfAdjointTolerance;
    spectrum.isNumericallyStarSelfAdjoint = isNumericallyStarSelfAdjoint;
    spectrum.normalityResidual = normalityResidual;
    spectrum.normalityTolerance = normalityTolerance;
    spectrum.isNumericallyNormal = isNumericallyNormal;
    spectrum.regularRepresentation = ...
        local_regular_representation_metadata(alg.m, rowCount, nBlades);

    if isNumericallyStarSelfAdjoint
        % The raw fields above remain for A.  The Hermitian fields describe
        % h=(A+A^star)/2.  Eigensolution uses an explicit Hermitian numerical
        % projection only to remove roundoff from the exact represented map.
        symmetrizedCoefficientTensor = 0.5 * (coefficientTensor + ...
            matrixStarCoefficientTensor);
        symmetrizedMap = clifford_matrix_left_regular_matrix( ...
            symmetrizedCoefficientTensor, alg);
        spectrum.hermitian = local_hermitian_analysis( ...
            blockLeftRegularMatrix, symmetrizedMap, ...
            symmetrizedCoefficientTensor, rowCount, nBlades, sigmaMax, ...
            eigenvalueClusterTolerance, positiveDefiniteTolerance);
    else
        spectrum.hermitian = local_empty_hermitian_analysis();
        spectrum.hermitian.positiveDefiniteTolerance = ...
            positiveDefiniteTolerance;
    end
end

function options = local_validate_options(options)
    if isempty(options)
        options = struct();
    end
    if ~(isstruct(options) && isscalar(options))
        error('clifford_matrix_left_regular_eigensystem:InvalidOptions', ...
            'OPTIONS must be a scalar structure.');
    end

    allowedFields = {'relativeTolerance', 'absoluteTolerance', ...
        'eigenvalueClusterTolerance', 'singularValueTolerance', ...
        'positiveDefiniteTolerance', 'selfAdjointTolerance', ...
        'normalityTolerance'};
    suppliedFields = fieldnames(options);
    for fieldNumber = 1:numel(suppliedFields)
        if ~any(strcmp(suppliedFields{fieldNumber}, allowedFields))
            error('clifford_matrix_left_regular_eigensystem:UnknownOption', ...
                'OPTIONS contains the unsupported field "%s".', ...
                suppliedFields{fieldNumber});
        end
    end

    normalized = struct();
    normalized.relativeTolerance = [];
    normalized.absoluteTolerance = 0;
    normalized.eigenvalueClusterTolerance = [];
    normalized.singularValueTolerance = [];
    normalized.positiveDefiniteTolerance = [];
    normalized.selfAdjointTolerance = [];
    normalized.normalityTolerance = [];
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
    normalized.eigenvalueClusterTolerance = ...
        local_validate_optional_tolerance( ...
        normalized.eigenvalueClusterTolerance, 'eigenvalueClusterTolerance');
    normalized.singularValueTolerance = local_validate_optional_tolerance( ...
        normalized.singularValueTolerance, 'singularValueTolerance');
    normalized.positiveDefiniteTolerance = ...
        local_validate_optional_tolerance( ...
        normalized.positiveDefiniteTolerance, 'positiveDefiniteTolerance');
    normalized.selfAdjointTolerance = local_validate_optional_tolerance( ...
        normalized.selfAdjointTolerance, 'selfAdjointTolerance');
    normalized.normalityTolerance = local_validate_optional_tolerance( ...
        normalized.normalityTolerance, 'normalityTolerance');
    options = normalized;
end

function value = local_validate_optional_tolerance(value, fieldName)
    if isempty(value)
        return;
    end
    if ~(isnumeric(value) && isreal(value) && isscalar(value) && ...
            isfinite(value) && value >= 0)
        error('clifford_matrix_left_regular_eigensystem:InvalidTolerance', ...
            ['OPTIONS.%s must be empty or one finite nonnegative ', ...
             'real scalar.'], fieldName);
    end
    value = double(value);
end

function tolerance = local_choose_tolerance(requestedTolerance, defaultTolerance)
    if isempty(requestedTolerance)
        tolerance = defaultTolerance;
    else
        tolerance = requestedTolerance;
    end
end

function vectors = local_normalize_columns(vectors)
    for column = 1:size(vectors, 2)
        columnNorm = norm(vectors(:, column));
        if columnNorm ~= 0
            vectors(:, column) = vectors(:, column) / columnNorm;
        end
    end
end

function residuals = local_eigen_residuals(matrix, values, vectors, sigmaMax)
    numberOfValues = numel(values);
    residuals = zeros(numberOfValues, 1);
    for valueNumber = 1:numberOfValues
        vector = vectors(:, valueNumber);
        residual = matrix * vector - values(valueNumber) * vector;
        denominator = (sigmaMax + abs(values(valueNumber))) * norm(vector);
        residuals(valueNumber) = local_relative_matrix_norm(residual, denominator);
    end
end

function residual = local_relative_tensor_difference(left, right)
    denominator = norm(left(:));
    if denominator == 0
        residual = norm(left(:) - right(:));
    else
        residual = norm(left(:) - right(:)) / denominator;
    end
end

function residual = local_relative_matrix_difference(left, right)
    denominator = norm(left, 'fro');
    if denominator == 0
        residual = norm(left - right, 'fro');
    else
        residual = norm(left - right, 'fro') / denominator;
    end
end

function residual = local_relative_matrix_norm(value, denominator)
    numerator = norm(value, 'fro');
    if denominator == 0
        residual = numerator;
    else
        residual = numerator / denominator;
    end
end

function clusters = local_cluster_values(values, tolerance)
% Greedy numerical clusters whose diameter is at most TOLERANCE.

    numberOfValues = numel(values);
    clusters = struct('indices', {}, 'multiplicity', {}, 'centroid', {}, ...
        'maxDeviation', {}, 'diameter', {});
    assigned = false(numberOfValues, 1);

    while any(~assigned)
        firstIndex = find(~assigned, 1, 'first');
        assigned(firstIndex) = true;
        component = firstIndex;
        changed = true;
        while changed
            changed = false;
            candidates = find(~assigned);
            for candidateNumber = 1:numel(candidates)
                candidate = candidates(candidateNumber);
                if all(abs(values(candidate) - values(component)) <= tolerance)
                    assigned(candidate) = true;
                    component = [component; candidate]; %#ok<AGROW>
                    changed = true;
                end
            end
        end

        component = sort(component);
        componentValues = values(component);
        centroid = mean(componentValues);
        maxDeviation = max(abs(componentValues - centroid));
        diameter = 0;
        for valueNumber = 1:numel(componentValues)
            diameter = max(diameter, max(abs(componentValues - ...
                componentValues(valueNumber))));
        end

        cluster = struct();
        cluster.indices = component;
        cluster.multiplicity = numel(component);
        cluster.centroid = centroid;
        cluster.maxDeviation = maxDeviation;
        cluster.diameter = diameter;
        clusters(end + 1) = cluster; %#ok<AGROW>
    end
end

function tensor = local_decode_coefficient_tensor(blockMatrix, rowCount, nBlades)
    tensor = complex(zeros(rowCount, rowCount, nBlades));
    for row = 1:rowCount
        rowIndices = (row - 1) * nBlades + (1:nBlades);
        for column = 1:rowCount
            firstColumn = (column - 1) * nBlades + 1;
            tensor(row, column, :) = reshape( ...
                blockMatrix(rowIndices, firstColumn), [1, 1, nBlades]);
        end
    end
end

function starTensor = local_matrix_star(tensor, alg)
    rowCount = size(tensor, 1);
    nBlades = alg.nBlades;
    starTensor = complex(zeros(rowCount, rowCount, nBlades));
    for row = 1:rowCount
        for column = 1:rowCount
            coefficientVector = reshape(tensor(row, column, :), nBlades, 1);
            starTensor(column, row, :) = reshape( ...
                clifford_multivector_star(coefficientVector, alg), ...
                [1, 1, nBlades]);
        end
    end
end

function coefficients = local_unpack_right_eigenvectors( ...
        packedVectors, rowCount, nBlades)
    numberOfVectors = size(packedVectors, 2);
    coefficients = complex(zeros(rowCount, nBlades, numberOfVectors));
    for vectorNumber = 1:numberOfVectors
        coefficients(:, :, vectorNumber) = reshape( ...
            packedVectors(:, vectorNumber), nBlades, rowCount).';
    end
end

function metadata = local_regular_representation_metadata(m, rowCount, nBlades)
    cliffordIrreducibleBlockDimension = 2^floor(m / 2);
    matrixIrreducibleBlockDimension = ...
        rowCount * cliffordIrreducibleBlockDimension;
    metadata = struct();
    metadata.cliffordIrreducibleBlockDimension = ...
        cliffordIrreducibleBlockDimension;
    metadata.matrixIrreducibleBlockDimension = matrixIrreducibleBlockDimension;
    metadata.regularCopyMultiplicity = cliffordIrreducibleBlockDimension;
    metadata.fullDimension = rowCount * nBlades;
    metadata.isOddDimension = mod(m, 2) == 1;

    if metadata.isOddDimension
        metadata.minimalFaithfulDimension = 2 * matrixIrreducibleBlockDimension;
        metadata.structure = sprintf('M_%d(C) direct-sum M_%d(C)', ...
            matrixIrreducibleBlockDimension, matrixIrreducibleBlockDimension);
    else
        metadata.minimalFaithfulDimension = matrixIrreducibleBlockDimension;
        metadata.structure = sprintf('M_%d(C)', ...
            matrixIrreducibleBlockDimension);
    end
end

function hermitian = local_empty_hermitian_analysis()
    hermitian = struct();
    hermitian.isAvailable = false;
    hermitian.symmetrizedBlockLeftRegularMatrix = [];
    hermitian.numericallyHermitianMatrix = [];
    hermitian.symmetrizedCoefficientTensor = [];
    hermitian.symmetrizationResidual = [];
    hermitian.symmetrizedMapHermitianResidual = [];
    hermitian.eigenvalues = [];
    hermitian.packedRightEigenvectors = [];
    hermitian.rightEigenvectors = [];
    hermitian.rightEigenvectorCoefficients = [];
    hermitian.relativeEigenResiduals = [];
    hermitian.maxRelativeEigenResidual = [];
    hermitian.originalMatrixRelativeEigenResiduals = [];
    hermitian.maxOriginalMatrixRelativeEigenResidual = [];
    hermitian.unitarityResidual = [];
    hermitian.spectralReconstructionResidual = [];
    hermitian.originalMatrixReconstructionResidual = [];
    hermitian.eigenvalueClusters = struct('indices', {}, 'multiplicity', {}, ...
        'centroid', {}, 'maxDeviation', {}, 'diameter', {});
    hermitian.lambdaMin = [];
    hermitian.lambdaMax = [];
    hermitian.positiveDefiniteTolerance = [];
    hermitian.positiveDefinitenessStatus = 'not applicable';
    hermitian.isNumericallyPositiveDefinite = false;
    hermitian.positiveDefiniteConditionNumber2 = NaN;
    hermitian.inverseSquareRootOperatorNorm = NaN;
end

function hermitian = local_hermitian_analysis(rawMatrix, symmetrizedMap, ...
        symmetrizedCoefficientTensor, rowCount, nBlades, sigmaMax, ...
        clusterTolerance, positiveDefiniteTolerance)

    numericallyHermitianMatrix = 0.5 * (symmetrizedMap + symmetrizedMap');
    complexDimension = size(numericallyHermitianMatrix, 1);
    [unitaryVectors, diagonalMatrix] = eig(numericallyHermitianMatrix);
    eigenvalues = real(diag(diagonalMatrix));
    [eigenvalues, order] = sort(eigenvalues);
    unitaryVectors = unitaryVectors(:, order);
    unitaryVectors = local_normalize_columns(unitaryVectors);
    relativeEigenResiduals = local_eigen_residuals(numericallyHermitianMatrix, ...
        eigenvalues, unitaryVectors, sigmaMax);
    originalMatrixRelativeEigenResiduals = local_eigen_residuals(rawMatrix, ...
        eigenvalues, unitaryVectors, sigmaMax);

    hermitian = local_empty_hermitian_analysis();
    hermitian.isAvailable = true;
    hermitian.symmetrizedBlockLeftRegularMatrix = symmetrizedMap;
    hermitian.numericallyHermitianMatrix = numericallyHermitianMatrix;
    hermitian.symmetrizedCoefficientTensor = symmetrizedCoefficientTensor;
    hermitian.symmetrizationResidual = local_relative_matrix_difference( ...
        rawMatrix, symmetrizedMap);
    hermitian.symmetrizedMapHermitianResidual = ...
        local_relative_matrix_difference(symmetrizedMap, symmetrizedMap');
    hermitian.eigenvalues = eigenvalues;
    hermitian.packedRightEigenvectors = unitaryVectors;
    hermitian.rightEigenvectors = unitaryVectors;
    hermitian.rightEigenvectorCoefficients = ...
        local_unpack_right_eigenvectors(unitaryVectors, rowCount, nBlades);
    hermitian.relativeEigenResiduals = relativeEigenResiduals;
    hermitian.maxRelativeEigenResidual = max(relativeEigenResiduals);
    hermitian.originalMatrixRelativeEigenResiduals = ...
        originalMatrixRelativeEigenResiduals;
    hermitian.maxOriginalMatrixRelativeEigenResidual = ...
        max(originalMatrixRelativeEigenResiduals);
    hermitian.unitarityResidual = local_relative_matrix_difference( ...
        unitaryVectors' * unitaryVectors, eye(complexDimension));
    hermitian.spectralReconstructionResidual = ...
        local_relative_matrix_difference(numericallyHermitianMatrix, ...
        unitaryVectors * diag(eigenvalues) * unitaryVectors');
    hermitian.originalMatrixReconstructionResidual = ...
        local_relative_matrix_difference(rawMatrix, ...
        unitaryVectors * diag(eigenvalues) * unitaryVectors');
    hermitian.eigenvalueClusters = local_cluster_values( ...
        eigenvalues, clusterTolerance);
    hermitian.lambdaMin = eigenvalues(1);
    hermitian.lambdaMax = eigenvalues(end);
    hermitian.positiveDefiniteTolerance = positiveDefiniteTolerance;

    if hermitian.lambdaMin > positiveDefiniteTolerance
        hermitian.positiveDefinitenessStatus = 'positive definite';
        hermitian.isNumericallyPositiveDefinite = true;
        hermitian.positiveDefiniteConditionNumber2 = ...
            hermitian.lambdaMax / hermitian.lambdaMin;
        hermitian.inverseSquareRootOperatorNorm = ...
            1 / sqrt(hermitian.lambdaMin);
    elseif hermitian.lambdaMin < -positiveDefiniteTolerance
        hermitian.positiveDefinitenessStatus = 'not positive definite';
    else
        hermitian.positiveDefinitenessStatus = ...
            'near the positive-semidefinite boundary';
    end
end
