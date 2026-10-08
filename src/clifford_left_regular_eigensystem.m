function spectrum = clifford_left_regular_eigensystem(a, alg, options)
%CLIFFORD_LEFT_REGULAR_EIGENSYSTEM Spectrum of one Clifford multivector.
%
%   SPECTRUM = CLIFFORD_LEFT_REGULAR_EIGENSYSTEM(A, ALG) constructs the
%   faithful left-regular matrix L_A and computes its ordinary complex
%   eigenvalues and right eigenvectors.  If V is SPECTRUM.rightEigenvectors
%   and LAMBDA is SPECTRUM.eigenvalues, then the j-th coefficient column V(:,j)
%   represents a multivector X_j satisfying
%
%       A*X_j = LAMBDA(j)*X_j.
%
%   The scalar LAMBDA(j) is an ordinary central complex scalar.  Thus this
%   routine is a spectrum utility for one multivector through its faithful
%   regular representation; it is not yet an eigensolver for a matrix whose
%   entries are Clifford multivectors.
%
%   The returned eigenvalue values agree with the algebra spectrum:
%
%       z belongs to spectrum(A)  <=>  z*1 - A is not invertible
%                                <=>  z*I - L_A is singular.
%
%   The regular representation can, however, force repeated eigenvalues and
%   gives noncanonical coefficient eigenvectors.  See the accompanying README
%   for the even- and odd-dimensional multiplicities.
%
%   SPECTRUM = CLIFFORD_LEFT_REGULAR_EIGENSYSTEM(A, ALG, OPTIONS) accepts a
%   scalar structure with any of these optional fields:
%
%       relativeTolerance           relative numerical resolution; default
%                                   is 100*ALG.nBlades*eps.
%       absoluteTolerance           absolute numerical resolution; default 0.
%       eigenvalueClusterTolerance  maximum diameter of each greedy numerical
%                                   cluster; default is ABS + REL*sigmaMax.
%       singularValueTolerance      absolute numerical rank cutoff; default
%                                   is ABS + REL*sigmaMax.
%       positiveDefiniteTolerance   absolute cutoff for lambda_min in the
%                                   self-star branch; same default.
%       selfAdjointTolerance        relative star-self-adjoint test; default
%                                   is REL.
%       normalityTolerance          relative normality test; default is REL.
%       computeSpectralProjectors   logical flag, default false.  When true,
%                                   form projectors for complete numerical
%                                   Hermitian eigenvalue clusters.
%       computeInverseSquareRoot    logical flag, default false.  When true,
%                                   form the inverse square root only for a
%                                   numerically positive-definite self-star A.
%
%   Tolerances are diagnostics, not modifications of the supplied data.  In
%   particular, eigenvalue clustering never discards or deduplicates the raw
%   eigenvalues.  For nonnormal A, use sigmaMin and conditionNumber2 rather
%   than min(abs(eigenvalues)) to assess invertibility or stability.

    if nargin < 3
        options = struct();
    end
    options = local_validate_options(options);

    % This call also validates A and ALG and puts A into a column vector.
    leftRegular = clifford_left_regular_matrix(a, alg);
    coefficientVector = leftRegular(:, 1);
    nBlades = alg.nBlades;

    singularValues = svd(leftRegular);
    sigmaMax = singularValues(1);
    sigmaMin = singularValues(end);

    if isempty(options.relativeTolerance)
        relativeTolerance = 100 * nBlades * eps;
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

    [rightEigenvectors, eigenvalueMatrix] = eig(leftRegular);
    eigenvalues = diag(eigenvalueMatrix);
    rightEigenvectors = local_normalize_columns(rightEigenvectors);
    relativeEigenResiduals = local_eigen_residuals(leftRegular, ...
        eigenvalues, rightEigenvectors, sigmaMax);

    numericalRank = sum(singularValues > singularValueTolerance);
    isNumericallySingular = numericalRank < nBlades;
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

    aStar = clifford_multivector_star(coefficientVector, alg);
    starSelfAdjointResidual = local_relative_vector_difference( ...
        coefficientVector, aStar);
    hermitianMatrixResidual = local_relative_matrix_difference( ...
        leftRegular, leftRegular');

    normalityCommutator = leftRegular' * leftRegular - ...
        leftRegular * leftRegular';
    normalityResidual = local_relative_matrix_norm( ...
        normalityCommutator, norm(leftRegular, 'fro')^2);

    isNumericallyStarSelfAdjoint = ...
        starSelfAdjointResidual <= selfAdjointTolerance && ...
        hermitianMatrixResidual <= selfAdjointTolerance;
    isNumericallyNormal = normalityResidual <= normalityTolerance;
    leftRegularImageTolerance = max(relativeTolerance, 100 * nBlades * eps);

    spectrum = struct();
    spectrum.representation = 'left-regular';
    spectrum.m = alg.m;
    spectrum.nBlades = nBlades;
    spectrum.coefficientVector = coefficientVector;
    spectrum.leftRegularMatrix = leftRegular;
    spectrum.eigenvalues = eigenvalues;
    spectrum.rightEigenvectors = rightEigenvectors;
    spectrum.relativeEigenResiduals = relativeEigenResiduals;
    spectrum.maxRelativeEigenResidual = max(relativeEigenResiduals);
    spectrum.eigenvectorMatrixConditionNumber2 = cond(rightEigenvectors);
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
    spectrum.aStar = aStar;
    spectrum.starSelfAdjointResidual = starSelfAdjointResidual;
    spectrum.hermitianMatrixResidual = hermitianMatrixResidual;
    spectrum.selfAdjointTolerance = selfAdjointTolerance;
    spectrum.isNumericallyStarSelfAdjoint = isNumericallyStarSelfAdjoint;
    spectrum.normalityResidual = normalityResidual;
    spectrum.normalityTolerance = normalityTolerance;
    spectrum.isNumericallyNormal = isNumericallyNormal;
    spectrum.regularRepresentation = local_regular_representation_metadata(alg.m);

    if isNumericallyStarSelfAdjoint
        % For an approximately self-star input, the Hermitian fields below
        % describe h=(a+a^star)/2, while the top-level fields remain for a.
        symmetrizedCoefficientVector = 0.5 * (coefficientVector + aStar);
        spectrum.hermitian = local_hermitian_analysis(leftRegular, ...
            symmetrizedCoefficientVector, alg, sigmaMax, ...
            eigenvalueClusterTolerance, positiveDefiniteTolerance, ...
            leftRegularImageTolerance, options.computeSpectralProjectors, ...
            options.computeInverseSquareRoot);
    else
        spectrum.hermitian = local_empty_hermitian_analysis();
        spectrum.hermitian.positiveDefiniteTolerance = positiveDefiniteTolerance;
        spectrum.hermitian.leftRegularImageTolerance = ...
            leftRegularImageTolerance;
    end
end

function options = local_validate_options(options)
    if isempty(options)
        options = struct();
    end
    if ~(isstruct(options) && isscalar(options))
        error('clifford_left_regular_eigensystem:InvalidOptions', ...
            'OPTIONS must be a scalar structure.');
    end

    allowedFields = {'relativeTolerance', 'absoluteTolerance', ...
        'eigenvalueClusterTolerance', 'singularValueTolerance', ...
        'positiveDefiniteTolerance', 'selfAdjointTolerance', ...
        'normalityTolerance', 'computeSpectralProjectors', ...
        'computeInverseSquareRoot'};
    suppliedFields = fieldnames(options);
    for fieldNumber = 1:numel(suppliedFields)
        if ~any(strcmp(suppliedFields{fieldNumber}, allowedFields))
            error('clifford_left_regular_eigensystem:UnknownOption', ...
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
    normalized.computeSpectralProjectors = false;
    normalized.computeInverseSquareRoot = false;

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
            normalized.eigenvalueClusterTolerance, ...
            'eigenvalueClusterTolerance');
    normalized.singularValueTolerance = local_validate_optional_tolerance( ...
        normalized.singularValueTolerance, 'singularValueTolerance');
    normalized.positiveDefiniteTolerance = ...
        local_validate_optional_tolerance( ...
            normalized.positiveDefiniteTolerance, ...
            'positiveDefiniteTolerance');
    normalized.selfAdjointTolerance = local_validate_optional_tolerance( ...
        normalized.selfAdjointTolerance, 'selfAdjointTolerance');
    normalized.normalityTolerance = local_validate_optional_tolerance( ...
        normalized.normalityTolerance, 'normalityTolerance');
    normalized.computeSpectralProjectors = local_validate_logical_option( ...
        normalized.computeSpectralProjectors, 'computeSpectralProjectors');
    normalized.computeInverseSquareRoot = local_validate_logical_option( ...
        normalized.computeInverseSquareRoot, 'computeInverseSquareRoot');

    options = normalized;
end

function value = local_validate_optional_tolerance(value, fieldName)
    if isempty(value)
        return;
    end
    if ~(isnumeric(value) && isreal(value) && isscalar(value) && ...
            isfinite(value) && value >= 0)
        error('clifford_left_regular_eigensystem:InvalidTolerance', ...
            ['OPTIONS.%s must be empty or one finite nonnegative ', ...
             'real scalar.'], fieldName);
    end
    value = double(value);
end

function value = local_validate_logical_option(value, fieldName)
    isLogicalScalar = islogical(value) && isscalar(value);
    isBinaryNumeric = isnumeric(value) && isreal(value) && isscalar(value) && ...
        isfinite(value) && (value == 0 || value == 1);
    if ~(isLogicalScalar || isBinaryNumeric)
        error('clifford_left_regular_eigensystem:InvalidLogicalOption', ...
            'OPTIONS.%s must be a logical scalar or numeric 0 or 1.', ...
            fieldName);
    end
    value = logical(value);
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
        residuals(valueNumber) = local_relative_matrix_norm( ...
            residual, denominator);
    end
end

function residual = local_relative_vector_difference(left, right)
    denominator = norm(left);
    if denominator == 0
        residual = norm(left - right);
    else
        residual = norm(left - right) / denominator;
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

        % A candidate must be close to every value already in the group.
        % This deliberately avoids transitive chains that would create a
        % cluster wider than the requested tolerance.
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

function metadata = local_regular_representation_metadata(m)
    irreducibleDimension = 2^floor(m / 2);
    metadata = struct();
    metadata.irreducibleBlockDimension = irreducibleDimension;
    metadata.regularCopyMultiplicity = irreducibleDimension;

    if mod(m, 2) == 0
        metadata.isOddDimension = false;
        metadata.minimalFaithfulDimension = irreducibleDimension;
        metadata.structure = sprintf('M_%d(C)', irreducibleDimension);
    else
        metadata.isOddDimension = true;
        metadata.minimalFaithfulDimension = 2 * irreducibleDimension;
        metadata.structure = sprintf('M_%d(C) direct-sum M_%d(C)', ...
            irreducibleDimension, irreducibleDimension);
    end
end

function hermitian = local_empty_hermitian_analysis()
    hermitian = struct();
    hermitian.isAvailable = false;
    hermitian.symmetrizedMatrix = [];
    hermitian.symmetrizedCoefficientVector = [];
    hermitian.symmetrizationResidual = [];
    hermitian.eigenvalues = [];
    hermitian.rightEigenvectors = [];
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
    hermitian.inverseSquareRootAvailable = false;
    hermitian.inverseSquareRootIsInLeftRegularImage = false;
    hermitian.inverseSquareRoot = [];
    hermitian.inverseSquareRootCoefficientVector = [];
    hermitian.inverseSquareRootImageResidual = [];
    hermitian.inverseSquareRootResidual = [];
    hermitian.projectorsAvailable = false;
    hermitian.spectralProjectors = {};
    hermitian.spectralProjectorCoefficientVectors = {};
    hermitian.spectralProjectorImageResiduals = [];
    hermitian.spectralProjectorIsInLeftRegularImage = [];
    hermitian.spectralProjectorIdempotenceResiduals = [];
    hermitian.projectorCompletenessResidual = [];
    hermitian.leftRegularImageTolerance = [];
end

function hermitian = local_hermitian_analysis(leftRegular, ...
        symmetrizedCoefficientVector, alg, sigmaMax, clusterTolerance, ...
        positiveDefiniteTolerance, imageTolerance, computeProjectors, ...
        computeInverseSquareRoot)

    nBlades = alg.nBlades;
    identityMatrix = eye(nBlades);
    hermitianMatrix = clifford_left_regular_matrix( ...
        symmetrizedCoefficientVector, alg);
    [unitaryVectors, diagonalMatrix] = eig(hermitianMatrix);
    eigenvalues = real(diag(diagonalMatrix));
    [eigenvalues, order] = sort(eigenvalues);
    unitaryVectors = unitaryVectors(:, order);
    unitaryVectors = local_normalize_columns(unitaryVectors);
    relativeEigenResiduals = local_eigen_residuals(hermitianMatrix, ...
        eigenvalues, unitaryVectors, sigmaMax);
    originalMatrixRelativeEigenResiduals = local_eigen_residuals( ...
        leftRegular, eigenvalues, unitaryVectors, sigmaMax);

    hermitian = local_empty_hermitian_analysis();
    hermitian.isAvailable = true;
    hermitian.symmetrizedMatrix = hermitianMatrix;
    hermitian.symmetrizedCoefficientVector = symmetrizedCoefficientVector;
    hermitian.symmetrizationResidual = local_relative_matrix_difference( ...
        leftRegular, hermitianMatrix);
    hermitian.eigenvalues = eigenvalues;
    hermitian.rightEigenvectors = unitaryVectors;
    hermitian.relativeEigenResiduals = relativeEigenResiduals;
    hermitian.maxRelativeEigenResidual = max(relativeEigenResiduals);
    hermitian.originalMatrixRelativeEigenResiduals = ...
        originalMatrixRelativeEigenResiduals;
    hermitian.maxOriginalMatrixRelativeEigenResidual = ...
        max(originalMatrixRelativeEigenResiduals);
    hermitian.unitarityResidual = local_relative_matrix_difference( ...
        unitaryVectors' * unitaryVectors, identityMatrix);
    hermitian.spectralReconstructionResidual = ...
        local_relative_matrix_difference(hermitianMatrix, ...
        unitaryVectors * diag(eigenvalues) * unitaryVectors');
    hermitian.originalMatrixReconstructionResidual = ...
        local_relative_matrix_difference(leftRegular, ...
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

    hermitian.leftRegularImageTolerance = imageTolerance;

    if computeProjectors
        [projectors, coefficientVectors, imageResiduals, inImage, ...
                idempotenceResiduals, completenessResidual] = ...
            local_spectral_projectors(unitaryVectors, ...
            hermitian.eigenvalueClusters, alg, imageTolerance);
        hermitian.projectorsAvailable = true;
        hermitian.spectralProjectors = projectors;
        hermitian.spectralProjectorCoefficientVectors = coefficientVectors;
        hermitian.spectralProjectorImageResiduals = imageResiduals;
        hermitian.spectralProjectorIsInLeftRegularImage = inImage;
        hermitian.spectralProjectorIdempotenceResiduals = idempotenceResiduals;
        hermitian.projectorCompletenessResidual = completenessResidual;
    end

    if computeInverseSquareRoot && hermitian.isNumericallyPositiveDefinite
        inverseSquareRoot = unitaryVectors * ...
            diag(1 ./ sqrt(eigenvalues)) * unitaryVectors';
        inverseCoefficientVector = inverseSquareRoot(:, 1);
        reconstructedInverse = clifford_left_regular_matrix( ...
            inverseCoefficientVector, alg);
        inverseImageResidual = local_relative_matrix_difference( ...
            inverseSquareRoot, reconstructedInverse);

        hermitian.inverseSquareRootAvailable = true;
        hermitian.inverseSquareRoot = inverseSquareRoot;
        hermitian.inverseSquareRootResidual = local_relative_matrix_difference( ...
            inverseSquareRoot * hermitianMatrix * inverseSquareRoot, ...
            identityMatrix);
        hermitian.inverseSquareRootImageResidual = inverseImageResidual;
        if inverseImageResidual <= imageTolerance
            hermitian.inverseSquareRootCoefficientVector = ...
                inverseCoefficientVector;
            hermitian.inverseSquareRootIsInLeftRegularImage = true;
        end
    end
end

function [projectors, coefficientVectors, imageResiduals, inImage, ...
        idempotenceResiduals, completenessResidual] = ...
        local_spectral_projectors(unitaryVectors, clusters, alg, imageTolerance)

    nBlades = size(unitaryVectors, 1);
    numberOfClusters = numel(clusters);
    projectors = cell(1, numberOfClusters);
    coefficientVectors = cell(1, numberOfClusters);
    imageResiduals = zeros(numberOfClusters, 1);
    inImage = false(numberOfClusters, 1);
    idempotenceResiduals = zeros(numberOfClusters, 1);
    projectorSum = zeros(nBlades, nBlades);

    for clusterNumber = 1:numberOfClusters
        indices = clusters(clusterNumber).indices;
        basis = unitaryVectors(:, indices);
        projector = basis * basis';
        projectorSum = projectorSum + projector;
        coefficientVector = projector(:, 1);
        reconstructedProjector = clifford_left_regular_matrix( ...
            coefficientVector, alg);
        imageResidual = local_relative_matrix_difference( ...
            projector, reconstructedProjector);

        projectors{clusterNumber} = projector;
        imageResiduals(clusterNumber) = imageResidual;
        inImage(clusterNumber) = imageResidual <= imageTolerance;
        idempotenceResiduals(clusterNumber) = ...
            local_relative_matrix_difference(projector * projector, projector);
        if inImage(clusterNumber)
            coefficientVectors{clusterNumber} = coefficientVector;
        else
            coefficientVectors{clusterNumber} = [];
        end
    end

    completenessResidual = local_relative_matrix_difference( ...
        projectorSum, eye(nBlades));
end
