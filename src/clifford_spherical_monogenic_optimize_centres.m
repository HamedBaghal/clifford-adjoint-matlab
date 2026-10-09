function result = clifford_spherical_monogenic_optimize_centres( ...
        m, degree, pointCount, options)
%CLIFFORD_SPHERICAL_MONOGENIC_OPTIMIZE_CENTRES Multistart sphere optimizer.
%
%   RESULT = CLIFFORD_SPHERICAL_MONOGENIC_OPTIMIZE_CENTRES(M, DEGREE,
%   POINTCOUNT) minimizes the off-diagonal spherical-monogenic kernel energy
%
%       F = sum_{i<j} |K_degree(eta_i,eta_j)|_Cl^2
%
%   on the product sphere (S^(m-1))^POINTCOUNT by projected gradient descent
%   with Armijo backtracking.  The final point of every restart is evaluated
%   by the established full left-regular Gram-matrix eigensystem utility.
%
%   A positive stability requirement may be given with exactly one of
%
%       OPTIONS.epsilon
%       OPTIONS.relativeAmplificationLimit
%
%   The latter is converted to EPSILON=d/Amax^2, where
%   d=binomial(DEGREE+M-2,DEGREE) is the fixed diagonal level and
%   A_rel=sqrt(d/lambda_min).  With a target present, the selected output is
%   the lowest-F restart satisfying lambda_min >= epsilon within the documented
%   feasibility tolerance.  If no restart is feasible, RESULT.selected is
%   empty and RESULT.bestStabilityRun records the closest sampled candidate.
%
%   This is deliberately a stability-screened local optimizer: the projected
%   gradient uses the smooth F objective, then the exact full Gram spectrum
%   accepts or rejects final candidates.  It does not claim a global minimum
%   and does not replace a nonsmooth constrained global optimizer.
%
%   OPTIONS is optional and may contain:
%
%       epsilon                     [] (default), or a scalar in [0,d].
%       relativeAmplificationLimit  [] (default), or a finite scalar >= 1.
%       numRestarts                 positive integer; default 8.
%       maxIterations               nonnegative integer; default 800.
%       gradientTolerance           positive scalar; default 1e-9.
%       initialStepSize             positive scalar; default 1.
%       armijoCoefficient           scalar in (0,1); default 1e-4.
%       backtrackingFactor          scalar in (0,1); default 0.5.
%       minimumStepSize             positive scalar; default 1e-14.
%       feasibilityTolerance         [] (default) uses 100*eps*max(1,d), or
%                                   a finite nonnegative spectral tolerance.
%       maximumPackedDimension       positive integer or Inf; default 1024.
%                                   This protects against accidentally asking
%                                   for an impractically large full spectrum.
%       initialPoints               [] or POINTCOUNT-by-M real nonzero rows.
%                                   When supplied, it is the first restart and
%                                   is normalized before use.
%       randomSeed                  [] or one nonnegative integer.
%       anchorFirstPoint             true (default) holds the first row fixed.
%                                   This removes only global rotational
%                                   symmetry and does not exclude any Gram
%                                   configuration up to an O(m) rotation.
%       verbose                     false (default) or true.
%
%   Positive epsilon is impossible when POINTCOUNT exceeds d.  This follows
%   from rank Lambda(G)<=2^M*d.  The function rejects that case before doing
%   any numerical search.

    if nargin < 4
        options = struct();
    end
    m = local_validate_dimension(m);
    degree = local_validate_nonnegative_integer(degree, 'degree', ...
        'clifford_spherical_monogenic_optimize_centres:InvalidDegree');
    pointCount = local_validate_nonnegative_integer(pointCount, 'pointCount', ...
        'clifford_spherical_monogenic_optimize_centres:InvalidPointCount');
    if pointCount < 1
        error('clifford_spherical_monogenic_optimize_centres:InvalidPointCount', ...
            'POINTCOUNT must be a positive integer.');
    end

    options = local_validate_options(options);
    nBlades = 2^m;
    packedDimension = pointCount * nBlades;
    if packedDimension > options.maximumPackedDimension
        error('clifford_spherical_monogenic_optimize_centres:PackedDimensionTooLarge', ...
            ['The full left-regular Gram matrix would be %g-by-%g, exceeding ', ...
             'OPTIONS.maximumPackedDimension=%g. Increase that option only ', ...
             'when enough memory and eigensolver time are available.'], ...
            packedDimension, packedDimension, options.maximumPackedDimension);
    end
    alg = clifford_algebra(m);
    diagonalLevel = local_monogenic_dimension(m, degree);
    if ~isfinite(diagonalLevel)
        error('clifford_spherical_monogenic_optimize_centres:DegreeTooLarge', ...
            ['DEGREE produces a nonfinite monogenic dimension in double ', ...
             'precision. Use a smaller degree or a scaled formulation.']);
    end
    [epsilon, hasEpsilonTarget, epsilonSource, amplificationLimit] = ...
        local_resolve_epsilon(options, diagonalLevel);
    if isempty(options.feasibilityTolerance)
        feasibilityTolerance = 100 * eps * max(1, diagonalLevel);
    else
        feasibilityTolerance = options.feasibilityTolerance;
    end
    options.feasibilityTolerance = feasibilityTolerance;

    if epsilon > 0 && pointCount > diagonalLevel
        error('clifford_spherical_monogenic_optimize_centres:ImpossiblePositiveDefiniteTarget', ...
            ['A positive epsilon is impossible because POINTCOUNT=%d exceeds ', ...
             'the monogenic module rank d=%g. Use POINTCOUNT<=d or epsilon=0.'], ...
            pointCount, diagonalLevel);
    end

    initialPoints = local_prepare_initial_points( ...
        options.initialPoints, pointCount, m);
    savedRandomState = rng;
    restoreRandomState = onCleanup(@() rng(savedRandomState)); %#ok<NASGU>
    if ~isempty(options.randomSeed)
        rng(options.randomSeed, 'twister');
    end

    template = local_empty_run();
    runs = repmat(template, options.numRestarts, 1);
    objectiveOptions = struct('normalizePoints', false);
    for restart = 1:options.numRestarts
        if restart == 1 && ~isempty(initialPoints)
            startPoints = initialPoints;
        else
            startPoints = local_random_sphere_points(pointCount, m);
        end
        run = local_projected_gradient( ...
            startPoints, degree, alg, objectiveOptions, options);
        run.restart = restart;
        run = local_add_spectral_diagnostics( ...
            run, degree, alg, diagonalLevel, epsilon, hasEpsilonTarget, ...
            feasibilityTolerance);
        runs(restart) = run;
        if options.verbose
            fprintf(['Restart %d/%d: F=%.12g, lambda_min=%.12g, ', ...
                'iterations=%d, %s.\n'], restart, options.numRestarts, ...
                run.objectiveF, run.lambdaMin, run.iterations, run.exitFlag);
        end
    end

    bestObjectiveRunIndex = local_select_lowest_objective( ...
        runs, 1:numel(runs));
    bestStabilityRunIndex = local_select_largest_lambda( ...
        runs, 1:numel(runs));
    selectedRunIndex = local_select_run(runs, hasEpsilonTarget);
    selected = [];
    if ~isnan(selectedRunIndex)
        selected = runs(selectedRunIndex);
        [gramEntries, kernelInfo] = clifford_spherical_monogenic_gram( ...
            selected.points, degree, alg);
        spectrum = clifford_matrix_left_regular_eigensystem(gramEntries, alg);
        selected.gramEntries = gramEntries;
        selected.kernelInfo = kernelInfo;
        selected.spectrum = spectrum;
    end

    result = struct();
    result.representation = 'stability-screened-projected-gradient';
    result.m = m;
    result.nBlades = alg.nBlades;
    result.packedDimension = packedDimension;
    result.degree = degree;
    result.pointCount = pointCount;
    result.monogenicModuleRank = diagonalLevel;
    result.diagonalLevel = diagonalLevel;
    result.isSquareInterpolationEnsemble = pointCount == diagonalLevel;
    result.theoreticalPositiveDefinitenessPossible = pointCount <= diagonalLevel;
    result.epsilon = epsilon;
    result.hasEpsilonTarget = hasEpsilonTarget;
    result.epsilonSource = epsilonSource;
    result.epsilonFractionOfDiagonal = epsilon / diagonalLevel;
    result.relativeAmplificationLimit = amplificationLimit;
    result.feasibilityTolerance = feasibilityTolerance;
    result.frobeniusCertificateThreshold = ...
        (diagonalLevel - epsilon)^2 / (2 * alg.nBlades);
    result.optionsUsed = options;
    result.runs = runs;
    result.summary = local_summary_table(runs);
    result.bestObjectiveRunIndex = bestObjectiveRunIndex;
    result.bestObjectiveRun = runs(bestObjectiveRunIndex);
    result.bestStabilityRunIndex = bestStabilityRunIndex;
    result.bestStabilityRun = runs(bestStabilityRunIndex);
    result.selectedRunIndex = selectedRunIndex;
    result.selected = selected;
    result.isEpsilonAchieved = ~isempty(selected) && selected.isFeasible;
    result.isEpsilonAchievedWithinTolerance = ...
        ~isempty(selected) && selected.isFeasibleWithinTolerance;
    result.globalMinimumProved = false;
    if hasEpsilonTarget && isempty(selected)
        result.selectionRule = ...
            ['No sampled restart met epsilon; selected is empty and ', ...
             'bestStabilityRun is supplied for diagnosis.'];
    elseif hasEpsilonTarget
        result.selectionRule = ...
            ['Lowest F among numerically positive-definite restarts ', ...
             'satisfying lambdaMin >= epsilon within feasibilityTolerance.'];
    else
        result.selectionRule = 'Lowest F among all restarts.';
    end
end

function run = local_projected_gradient( ...
        startPoints, degree, alg, objectiveOptions, options)
    currentPoints = startPoints;
    [currentF, ~, currentGradient, currentInfo] = ...
        clifford_spherical_monogenic_objective( ...
        currentPoints, degree, alg, objectiveOptions);
    currentGradient = local_anchor_gradient(currentGradient, options);

    objectiveHistory = nan(options.maxIterations + 1, 1);
    gradientHistory = nan(options.maxIterations + 1, 1);
    stepHistory = nan(options.maxIterations, 1);
    objectiveHistory(1) = currentF;
    gradientNorm = norm(currentGradient, 'fro');
    gradientHistory(1) = gradientNorm;
    iterations = 0;

    if currentInfo.isConstantOnSphere
        exitFlag = 'constant-objective';
    elseif gradientNorm <= options.gradientTolerance
        exitFlag = 'gradient-tolerance';
    else
        exitFlag = 'maximum-iterations';
        for iteration = 1:options.maxIterations
            stepSize = options.initialStepSize;
            accepted = false;
            while stepSize >= options.minimumStepSize
                candidatePoints = local_normalize_rows( ...
                    currentPoints - stepSize * currentGradient);
                candidateF = clifford_spherical_monogenic_objective( ...
                    candidatePoints, degree, alg, objectiveOptions);
                if candidateF <= currentF - options.armijoCoefficient * ...
                        stepSize * gradientNorm^2
                    accepted = true;
                    break;
                end
                stepSize = options.backtrackingFactor * stepSize;
            end

            if ~accepted
                exitFlag = 'line-search-failed';
                break;
            end

            currentPoints = candidatePoints;
            currentF = candidateF;
            iterations = iteration;
            stepHistory(iteration) = stepSize;
            [currentF, ~, currentGradient] = ...
                clifford_spherical_monogenic_objective( ...
                currentPoints, degree, alg, objectiveOptions);
            currentGradient = local_anchor_gradient(currentGradient, options);
            gradientNorm = norm(currentGradient, 'fro');
            objectiveHistory(iteration + 1) = currentF;
            gradientHistory(iteration + 1) = gradientNorm;
            if gradientNorm <= options.gradientTolerance
                exitFlag = 'gradient-tolerance';
                break;
            end
        end
    end

    run = local_empty_run();
    run.initialPoints = startPoints;
    run.points = currentPoints;
    run.initialObjectiveF = objectiveHistory(1);
    run.objectiveF = currentF;
    run.iterations = iterations;
    run.exitFlag = exitFlag;
    run.finalGradientNorm = gradientNorm;
    run.objectiveHistory = objectiveHistory(1:(iterations + 1));
    run.gradientNormHistory = gradientHistory(1:(iterations + 1));
    run.stepSizeHistory = stepHistory(1:iterations);
    run.maxUnitSphereResidual = max(abs(sum(currentPoints.^2, 2) - 1));
end

function run = local_add_spectral_diagnostics( ...
        run, degree, alg, diagonalLevel, epsilon, hasEpsilonTarget, ...
        feasibilityTolerance)
    [entries, kernelInfo] = clifford_spherical_monogenic_gram( ...
        run.points, degree, alg);
    spectrum = clifford_matrix_left_regular_eigensystem(entries, alg);
    if ~(spectrum.hermitian.isAvailable && ...
            spectrum.isNumericallyStarSelfAdjoint)
        error('clifford_spherical_monogenic_optimize_centres:SpectralFailure', ...
            'The final Gram matrix was not numerically Hermitian as expected.');
    end

    lambdaMin = real(spectrum.hermitian.lambdaMin);
    lambdaMax = real(spectrum.hermitian.lambdaMax);
    spectralTolerance = spectrum.hermitian.positiveDefiniteTolerance;
    isNumericallyPositiveDefinite = ...
        spectrum.hermitian.isNumericallyPositiveDefinite;
    if isNumericallyPositiveDefinite
        conditionNumber2 = lambdaMax / lambdaMin;
        relativeAmplification = sqrt(diagonalLevel / lambdaMin);
    else
        conditionNumber2 = Inf;
        relativeAmplification = Inf;
    end
    if hasEpsilonTarget
        if epsilon > 0
            isStrictlyFeasible = isNumericallyPositiveDefinite && ...
                lambdaMin >= epsilon;
            isFeasibleWithinTolerance = isNumericallyPositiveDefinite && ...
                lambdaMin + feasibilityTolerance >= epsilon;
        else
            % epsilon=0 is a PSD baseline, not a positive-definite target.
            isStrictlyFeasible = true;
            isFeasibleWithinTolerance = true;
        end
    else
        isStrictlyFeasible = true;
        isFeasibleWithinTolerance = true;
    end

    run.lambdaMin = lambdaMin;
    run.lambdaMax = lambdaMax;
    run.conditionNumber2 = conditionNumber2;
    run.relativeAmplification = relativeAmplification;
    run.spectralTolerance = spectralTolerance;
    run.feasibilityTolerance = feasibilityTolerance;
    run.epsilonMargin = lambdaMin - epsilon;
    run.isNumericallyPositiveDefinite = isNumericallyPositiveDefinite;
    run.isFeasible = isStrictlyFeasible;
    run.isStrictlyFeasible = isStrictlyFeasible;
    run.isFeasibleWithinTolerance = isFeasibleWithinTolerance;
    run.frobeniusCertificateLowerBound = diagonalLevel - ...
        sqrt(2 * alg.nBlades * run.objectiveF);
    run.frobeniusCertificateMeetsEpsilon = ...
        run.frobeniusCertificateLowerBound >= epsilon;
    run.fullRegularFrobeniusEnergy = ...
        kernelInfo.fullRegularFrobeniusEnergy;
end

function selectedRunIndex = local_select_run(runs, hasEpsilonTarget)
    if hasEpsilonTarget
        feasibleIndices = find([runs.isFeasibleWithinTolerance]);
    else
        feasibleIndices = 1:numel(runs);
    end
    if ~isempty(feasibleIndices)
        selectedRunIndex = local_select_lowest_objective(runs, feasibleIndices);
        return;
    end
    selectedRunIndex = NaN;
end

function selectedRunIndex = local_select_lowest_objective(runs, indices)
    values = [runs(indices).objectiveF];
    minimum = min(values);
    tied = indices(abs(values - minimum) <= ...
        100 * eps(max(1, abs(minimum))));
    if numel(tied) == 1
        selectedRunIndex = tied;
    else
        [~, position] = max([runs(tied).lambdaMin]);
        selectedRunIndex = tied(position);
    end
end

function selectedRunIndex = local_select_largest_lambda(runs, indices)
    lambdaValues = [runs(indices).lambdaMin];
    largest = max(lambdaValues);
    tied = indices(abs(lambdaValues - largest) <= ...
        100 * eps(max(1, abs(largest))));
    if numel(tied) == 1
        selectedRunIndex = tied;
    else
        [~, position] = min([runs(tied).objectiveF]);
        selectedRunIndex = tied(position);
    end
end

function summary = local_summary_table(runs)
    restart = [runs.restart].';
    initialF = [runs.initialObjectiveF].';
    F = [runs.objectiveF].';
    lambdaMin = [runs.lambdaMin].';
    lambdaMax = [runs.lambdaMax].';
    kappa2 = [runs.conditionNumber2].';
    relativeAmplification = [runs.relativeAmplification].';
    certificateLowerBound = [runs.frobeniusCertificateLowerBound].';
    iterations = [runs.iterations].';
    gradientNorm = [runs.finalGradientNorm].';
    epsilonMargin = [runs.epsilonMargin].';
    feasible = [runs.isFeasible].';
    strictlyFeasible = [runs.isStrictlyFeasible].';
    feasibleWithinTolerance = [runs.isFeasibleWithinTolerance].';
    exitFlag = {runs.exitFlag}.';
    summary = table(restart, initialF, F, lambdaMin, lambdaMax, kappa2, ...
        relativeAmplification, certificateLowerBound, iterations, ...
        gradientNorm, epsilonMargin, feasible, strictlyFeasible, ...
        feasibleWithinTolerance, exitFlag, ...
        'VariableNames', {'Restart', 'InitialF', 'F', 'lambdaMin', ...
        'lambdaMax', 'kappa2', 'relativeAmplification', ...
        'certificateLowerBound', 'iterations', 'gradientNorm', ...
        'epsilonMargin', 'feasible', 'strictlyFeasible', ...
        'feasibleWithinTolerance', 'exitFlag'});
end

function points = local_random_sphere_points(pointCount, m)
    points = local_normalize_rows(randn(pointCount, m));
end

function points = local_normalize_rows(points)
    rowNorms = zeros(size(points, 1), 1);
    for row = 1:size(points, 1)
        rowNorms(row) = norm(points(row, :));
    end
    if any(~isfinite(rowNorms)) || any(rowNorms == 0)
        error('clifford_spherical_monogenic_optimize_centres:ZeroPoint', ...
            ['A sphere retraction encountered a zero or nonfinite-norm ', ...
             'row.']);
    end
    points = bsxfun(@rdivide, points, rowNorms);
end

function gradient = local_anchor_gradient(gradient, options)
    if options.anchorFirstPoint
        gradient(1, :) = 0;
    end
end

function initialPoints = local_prepare_initial_points( ...
        initialPoints, pointCount, m)
    if isempty(initialPoints)
        initialPoints = [];
        return;
    end
    if ~(isnumeric(initialPoints) && isreal(initialPoints) && ...
            ismatrix(initialPoints) && size(initialPoints, 1) == pointCount && ...
            size(initialPoints, 2) == m && all(isfinite(initialPoints(:))))
        error('clifford_spherical_monogenic_optimize_centres:InvalidInitialPoints', ...
            ['OPTIONS.initialPoints must be a finite real POINTCOUNT-by-M ', ...
             'matrix.']);
    end
    initialPoints = double(initialPoints);
    initialPoints = local_normalize_rows(initialPoints);
end

function [epsilon, hasTarget, source, amplificationLimit] = ...
        local_resolve_epsilon(options, diagonalLevel)
    hasDirect = ~isempty(options.epsilon);
    hasAmplification = ~isempty(options.relativeAmplificationLimit);
    if hasDirect && hasAmplification
        error('clifford_spherical_monogenic_optimize_centres:AmbiguousEpsilon', ...
            ['Specify either OPTIONS.epsilon or ', ...
             'OPTIONS.relativeAmplificationLimit, not both.']);
    end
    if hasDirect
        epsilon = options.epsilon;
        hasTarget = true;
        source = 'direct-epsilon';
        if epsilon == 0
            amplificationLimit = Inf;
        else
            amplificationLimit = sqrt(diagonalLevel / epsilon);
        end
    elseif hasAmplification
        amplificationLimit = options.relativeAmplificationLimit;
        epsilon = diagonalLevel / amplificationLimit^2;
        if ~(isfinite(epsilon) && epsilon > 0)
            error('clifford_spherical_monogenic_optimize_centres:AmplificationLimitTooLarge', ...
                ['OPTIONS.relativeAmplificationLimit is too large to ', ...
                 'produce a positive representable epsilon in double precision.']);
        end
        hasTarget = true;
        source = 'relative-amplification-limit';
    else
        epsilon = 0;
        hasTarget = false;
        source = 'none';
        amplificationLimit = [];
    end
    if epsilon > diagonalLevel + 100 * eps(max(1, diagonalLevel))
        error('clifford_spherical_monogenic_optimize_centres:EpsilonExceedsDiagonal', ...
            ['EPSILON cannot exceed the fixed diagonal level d=%g. ', ...
             'Such a spectral margin is impossible.'], diagonalLevel);
    end
    epsilon = min(epsilon, diagonalLevel);
end

function options = local_validate_options(options)
    if isempty(options)
        options = struct();
    end
    if ~(isstruct(options) && isscalar(options))
        error('clifford_spherical_monogenic_optimize_centres:InvalidOptions', ...
            'OPTIONS must be a scalar structure.');
    end

    allowedFields = {'epsilon', 'relativeAmplificationLimit', 'numRestarts', ...
        'maxIterations', 'gradientTolerance', 'initialStepSize', ...
        'armijoCoefficient', 'backtrackingFactor', 'minimumStepSize', ...
        'feasibilityTolerance', 'maximumPackedDimension', ...
        'initialPoints', 'randomSeed', 'anchorFirstPoint', 'verbose'};
    suppliedFields = fieldnames(options);
    for fieldNumber = 1:numel(suppliedFields)
        if ~any(strcmp(suppliedFields{fieldNumber}, allowedFields))
            error('clifford_spherical_monogenic_optimize_centres:UnknownOption', ...
                'OPTIONS contains the unsupported field "%s".', ...
                suppliedFields{fieldNumber});
        end
    end

    normalized = struct();
    normalized.epsilon = [];
    normalized.relativeAmplificationLimit = [];
    normalized.numRestarts = 8;
    normalized.maxIterations = 800;
    normalized.gradientTolerance = 1e-9;
    normalized.initialStepSize = 1;
    normalized.armijoCoefficient = 1e-4;
    normalized.backtrackingFactor = 0.5;
    normalized.minimumStepSize = 1e-14;
    normalized.feasibilityTolerance = [];
    normalized.maximumPackedDimension = 1024;
    normalized.initialPoints = [];
    normalized.randomSeed = [];
    normalized.anchorFirstPoint = true;
    normalized.verbose = false;
    for fieldNumber = 1:numel(suppliedFields)
        fieldName = suppliedFields{fieldNumber};
        normalized.(fieldName) = options.(fieldName);
    end

    normalized.epsilon = local_validate_optional_nonnegative_scalar( ...
        normalized.epsilon, 'epsilon');
    normalized.relativeAmplificationLimit = ...
        local_validate_optional_amplification( ...
        normalized.relativeAmplificationLimit, 'relativeAmplificationLimit');
    normalized.numRestarts = local_validate_positive_integer( ...
        normalized.numRestarts, 'numRestarts');
    normalized.maxIterations = local_validate_nonnegative_integer( ...
        normalized.maxIterations, 'maxIterations', ...
        'clifford_spherical_monogenic_optimize_centres:InvalidOption');
    normalized.gradientTolerance = local_validate_positive_scalar( ...
        normalized.gradientTolerance, 'gradientTolerance');
    normalized.initialStepSize = local_validate_positive_scalar( ...
        normalized.initialStepSize, 'initialStepSize');
    normalized.armijoCoefficient = local_validate_open_unit_scalar( ...
        normalized.armijoCoefficient, 'armijoCoefficient');
    normalized.backtrackingFactor = local_validate_open_unit_scalar( ...
        normalized.backtrackingFactor, 'backtrackingFactor');
    normalized.minimumStepSize = local_validate_positive_scalar( ...
        normalized.minimumStepSize, 'minimumStepSize');
    if normalized.minimumStepSize > normalized.initialStepSize
        error('clifford_spherical_monogenic_optimize_centres:InvalidOption', ...
            'OPTIONS.minimumStepSize cannot exceed OPTIONS.initialStepSize.');
    end
    normalized.randomSeed = local_validate_optional_rng_seed( ...
        normalized.randomSeed, 'randomSeed');
    normalized.feasibilityTolerance = local_validate_optional_nonnegative_scalar( ...
        normalized.feasibilityTolerance, 'feasibilityTolerance');
    normalized.maximumPackedDimension = local_validate_packed_dimension( ...
        normalized.maximumPackedDimension, 'maximumPackedDimension');
    normalized.anchorFirstPoint = local_validate_boolean( ...
        normalized.anchorFirstPoint, 'anchorFirstPoint');
    normalized.verbose = local_validate_boolean(normalized.verbose, 'verbose');
    options = normalized;
end

function m = local_validate_dimension(m)
    m = local_validate_nonnegative_integer(m, 'm', ...
        'clifford_spherical_monogenic_optimize_centres:InvalidDimension');
    if m < 2 || m > 52
        error('clifford_spherical_monogenic_optimize_centres:UnsupportedDimension', ...
            'M must be an integer between 2 and 52.');
    end
end

function value = local_validate_nonnegative_integer(value, fieldName, identifier)
    if ~(isnumeric(value) && isreal(value) && isscalar(value) && ...
            isfinite(value) && value == floor(value) && value >= 0)
        error(identifier, '%s must be one finite nonnegative integer.', upper(fieldName));
    end
    value = double(value);
end

function value = local_validate_positive_integer(value, fieldName)
    value = local_validate_nonnegative_integer(value, fieldName, ...
        'clifford_spherical_monogenic_optimize_centres:InvalidOption');
    if value < 1
        error('clifford_spherical_monogenic_optimize_centres:InvalidOption', ...
            'OPTIONS.%s must be a positive integer.', fieldName);
    end
end

function value = local_validate_optional_nonnegative_integer(value, fieldName)
    if isempty(value)
        return;
    end
    value = local_validate_nonnegative_integer(value, fieldName, ...
        'clifford_spherical_monogenic_optimize_centres:InvalidOption');
end

function value = local_validate_optional_rng_seed(value, fieldName)
    if isempty(value)
        return;
    end
    value = local_validate_nonnegative_integer(value, fieldName, ...
        'clifford_spherical_monogenic_optimize_centres:InvalidOption');
    if value > 2^32 - 1
        error('clifford_spherical_monogenic_optimize_centres:InvalidOption', ...
            ['OPTIONS.%s must be an integer between 0 and 2^32-1 for ', ...
             'the twister random-number generator.'], fieldName);
    end
end

function value = local_validate_optional_nonnegative_scalar(value, fieldName)
    if isempty(value)
        return;
    end
    if ~(isnumeric(value) && isreal(value) && isscalar(value) && ...
            isfinite(value) && value >= 0)
        error('clifford_spherical_monogenic_optimize_centres:InvalidOption', ...
            'OPTIONS.%s must be empty or one finite nonnegative scalar.', fieldName);
    end
    value = double(value);
end

function value = local_validate_packed_dimension(value, fieldName)
    if isnumeric(value) && isreal(value) && isscalar(value) && ...
            isinf(value) && value > 0
        return;
    end
    value = local_validate_positive_integer(value, fieldName);
end

function value = local_validate_optional_amplification(value, fieldName)
    if isempty(value)
        return;
    end
    if ~(isnumeric(value) && isreal(value) && isscalar(value) && ...
            isfinite(value) && value >= 1)
        error('clifford_spherical_monogenic_optimize_centres:InvalidOption', ...
            ['OPTIONS.%s must be empty or one finite scalar at least 1. ', ...
             'A relative amplification bound below 1 is impossible.'], fieldName);
    end
    value = double(value);
end

function value = local_validate_positive_scalar(value, fieldName)
    if ~(isnumeric(value) && isreal(value) && isscalar(value) && ...
            isfinite(value) && value > 0)
        error('clifford_spherical_monogenic_optimize_centres:InvalidOption', ...
            'OPTIONS.%s must be one finite positive scalar.', fieldName);
    end
    value = double(value);
end

function value = local_validate_open_unit_scalar(value, fieldName)
    if ~(isnumeric(value) && isreal(value) && isscalar(value) && ...
            isfinite(value) && value > 0 && value < 1)
        error('clifford_spherical_monogenic_optimize_centres:InvalidOption', ...
            'OPTIONS.%s must be one finite scalar strictly between 0 and 1.', ...
            fieldName);
    end
    value = double(value);
end

function value = local_validate_boolean(value, fieldName)
    isLogical = islogical(value) && isscalar(value);
    isZeroOne = isnumeric(value) && isreal(value) && isscalar(value) && ...
        isfinite(value) && (value == 0 || value == 1);
    if ~(isLogical || isZeroOne)
        error('clifford_spherical_monogenic_optimize_centres:InvalidOption', ...
            'OPTIONS.%s must be one logical value or numeric 0/1.', fieldName);
    end
    value = logical(value);
end

function dimension = local_monogenic_dimension(m, degree)
    dimension = nchoosek(degree + m - 2, degree);
end

function run = local_empty_run()
    run = struct('restart', NaN, 'initialPoints', [], 'points', [], ...
        'initialObjectiveF', NaN, 'objectiveF', NaN, 'iterations', NaN, ...
        'exitFlag', '', 'finalGradientNorm', NaN, 'objectiveHistory', [], ...
        'gradientNormHistory', [], 'stepSizeHistory', [], ...
        'maxUnitSphereResidual', NaN, 'lambdaMin', NaN, 'lambdaMax', NaN, ...
        'conditionNumber2', NaN, 'relativeAmplification', NaN, ...
        'spectralTolerance', NaN, 'isNumericallyPositiveDefinite', false, ...
        'feasibilityTolerance', NaN, 'epsilonMargin', NaN, ...
        'isFeasible', false, 'isStrictlyFeasible', false, ...
        'isFeasibleWithinTolerance', false, ...
        'frobeniusCertificateLowerBound', NaN, ...
        'frobeniusCertificateMeetsEpsilon', false, ...
        'fullRegularFrobeniusEnergy', NaN);
end
