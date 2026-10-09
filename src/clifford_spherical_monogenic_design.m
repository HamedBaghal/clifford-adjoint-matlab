function design = clifford_spherical_monogenic_design(m, degree, options)
%CLIFFORD_SPHERICAL_MONOGENIC_DESIGN Build a seeded spherical kernel system.
%
%   DESIGN = CLIFFORD_SPHERICAL_MONOGENIC_DESIGN(M, DEGREE) constructs a
%   reproducible centre-selection problem for the degree-DEGREE
%   spherical-monogenic kernel on S^(M-1).  It chooses
%
%       r = binomial(DEGREE+M-2, DEGREE)
%
%   centres by default, builds a kernel-aware initial configuration, then
%   calls CLIFFORD_SPHERICAL_MONOGENIC_OPTIMIZE_CENTRES.  The returned DESIGN
%   packages the selected centres, Gram tensor, full left-regular spectrum,
%   and the complete optimizer report.
%
%   DESIGN = CLIFFORD_SPHERICAL_MONOGENIC_DESIGN(M, DEGREE, OPTIONS) accepts
%   the following additional design options:
%
%       pointCount            [] (default) uses the diagonal level d_m,k;
%                             otherwise one positive integer.
%       seedStrategy          'hybrid' (default), 'random',
%                             'kernel-greedy', 'user', or 'continuation'.
%       greedyCandidateCount  positive integer; default 512.  At every
%                             greedy stage this many random sphere points
%                             are screened by their exact added F energy.
%       initialPoints         required for seedStrategy='user'.
%       continuationPoints    required for seedStrategy='continuation'.
%       continuationJitter    nonnegative scalar; default 0.  A tangent
%                             perturbation of this size is applied to a
%                             continuation seed before optimization.
%       randomSeed            [] or one integer in [0,2^32-1].  It makes
%                             both the greedy candidate pool and random
%                             restarts reproducible without changing the
%                             caller's random-number state.
%
%   All other OPTIONS fields are passed directly to
%   CLIFFORD_SPHERICAL_MONOGENIC_OPTIMIZE_CENTRES:
%
%       epsilon, relativeAmplificationLimit, numRestarts, maxIterations,
%       gradientTolerance, initialStepSize, armijoCoefficient,
%       backtrackingFactor, minimumStepSize, feasibilityTolerance,
%       maximumPackedDimension, anchorFirstPoint, verbose.
%
%   Seed-policy meaning:
%
%       hybrid          one kernel-greedy start, then independent random
%                       starts supplied by the optimizer.
%       random          every optimizer start is random.
%       kernel-greedy   one kernel-greedy start only; NUMRESTARTS must be 1.
%       user            INITIALPOINTS is restart 1; later starts, if any,
%                       are random.
%       continuation    CONTINUATIONPOINTS is restart 1; later starts, if
%                       any, are random.
%
%   The kernel-greedy stage is deliberately based on the actual pair energy
%
%       q_m,k(<x,y>) = |K_degree(x,y)|_Cl^2,
%
%   rather than ordinary Euclidean point separation.  Therefore it is not a
%   claim that a regular polyhedron or a conventional spherical code is
%   universally optimal for this kernel.
%
%   DESIGN constructs a numerical kernel/interpolation system.  It does not
%   create a symbolic list of an Appell basis of all spherical monogenics, and
%   DESIGN.globalMinimumProved is always false.

    if nargin < 3
        options = struct();
    end
    m = local_validate_dimension(m);
    degree = local_validate_nonnegative_integer(degree, 'degree', ...
        'clifford_spherical_monogenic_design:InvalidDegree');
    options = local_validate_options(options);

    diagonalLevel = local_monogenic_dimension(m, degree);
    if ~isfinite(diagonalLevel)
        error('clifford_spherical_monogenic_design:DegreeTooLarge', ...
            ['DEGREE produces a nonfinite monogenic dimension in double ', ...
             'precision. Use a smaller degree or a scaled formulation.']);
    end
    if isempty(options.pointCount)
        pointCount = diagonalLevel;
    else
        pointCount = options.pointCount;
    end
    if ~isempty(options.epsilon) && options.epsilon > ...
            diagonalLevel + 100 * eps(max(1, diagonalLevel))
        error('clifford_spherical_monogenic_design:EpsilonExceedsDiagonal', ...
            ['EPSILON cannot exceed the fixed diagonal level d=%g. ', ...
             'Such a spectral margin is impossible.'], diagonalLevel);
    end
    if local_has_positive_stability_target(options) && ...
            pointCount > diagonalLevel
        error('clifford_spherical_monogenic_design:ImpossiblePositiveDefiniteTarget', ...
            ['A positive epsilon is impossible because POINTCOUNT=%d exceeds ', ...
             'the monogenic module rank d=%g. Use POINTCOUNT<=d or remove ', ...
             'the positive stability target.'], pointCount, diagonalLevel);
    end

    nBlades = 2^m;
    packedDimension = pointCount * nBlades;
    if packedDimension > options.maximumPackedDimension
        error('clifford_spherical_monogenic_design:PackedDimensionTooLarge', ...
            ['The full left-regular Gram matrix would be %g-by-%g, exceeding ', ...
             'OPTIONS.maximumPackedDimension=%g. Increase that option only ', ...
             'when enough memory and eigensolver time are available.'], ...
            packedDimension, packedDimension, options.maximumPackedDimension);
    end

    alg = clifford_algebra(m);
    savedRandomState = rng;
    restoreRandomState = onCleanup(@() rng(savedRandomState)); %#ok<NASGU>
    if ~isempty(options.randomSeed)
        rng(options.randomSeed, 'twister');
    end

    [seedPoints, seed] = local_choose_seed( ...
        m, degree, pointCount, alg, options);
    optimizerOptions = local_optimizer_options(options, seedPoints, seed);
    seed.optimizerRandomSeed = optimizerOptions.randomSeed;
    seed.optimizerNumRestarts = optimizerOptions.numRestarts;
    seed.randomRestartCount = optimizerOptions.numRestarts - ...
        double(~isempty(seedPoints));

    optimizerResult = clifford_spherical_monogenic_optimize_centres( ...
        m, degree, pointCount, optimizerOptions);
    design = local_assemble_design( ...
        m, degree, pointCount, diagonalLevel, alg, options, seed, ...
        optimizerOptions, optimizerResult);
end

function [seedPoints, seed] = local_choose_seed( ...
        m, degree, pointCount, alg, options)
    seed = local_empty_seed(options, pointCount, m);
    seed.isFlatObjective = (m == 2) || (degree == 0);
    switch options.seedStrategy
        case 'random'
            seedPoints = [];
            seed.source = 'independent-random-restarts';
            seed.description = ...
                'All optimizer starts are independent normalized Gaussian points.';

        case {'hybrid', 'kernel-greedy'}
            if m == 2 || degree == 0
                [seedPoints, greedyTrace] = local_flat_objective_seed( ...
                    pointCount, m);
                isKernelAware = false;
                greedySkippedReason = ...
                    'The spherical-monogenic pair energy is constant.';
            else
                [seedPoints, greedyTrace] = local_kernel_greedy_seed( ...
                    m, degree, pointCount, alg, options.greedyCandidateCount);
                isKernelAware = true;
                greedySkippedReason = '';
            end
            [seedObjectiveF, ~, ~, objectiveInfo] = ...
                clifford_spherical_monogenic_objective( ...
                seedPoints, degree, alg);
            if isKernelAware
                seed.source = 'kernel-greedy';
            else
                seed.source = 'flat-objective-fallback';
            end
            seed.isKernelAware = isKernelAware;
            seed.isFlatObjective = ~isKernelAware;
            seed.greedySkippedReason = greedySkippedReason;
            seed.firstPointPolicy = 'canonical-e1';
            seed.points = seedPoints;
            seed.initialPoints = seedPoints;
            seed.canonicalAnchor = [1, zeros(1, m - 1)];
            seed.inputPointNorms = ones(pointCount, 1);
            seed.initialObjectiveF = seedObjectiveF;
            seed.initialObjectiveInfo = local_compact_objective_info(objectiveInfo);
            seed.greedyTrace = greedyTrace;
            if strcmp(options.seedStrategy, 'hybrid')
                if isKernelAware
                    seed.description = ...
                        ['Restart 1 is kernel-greedy; later optimizer starts ', ...
                         'are independent random sphere points.'];
                else
                    seed.description = ...
                        ['The objective is flat, so restart 1 is a canonical ', ...
                         'random fallback rather than an energy-optimized seed.'];
                end
            else
                if isKernelAware
                    seed.description = ...
                        'One kernel-greedy start only; no random restart is used.';
                else
                    seed.description = ...
                        ['The objective is flat, so one canonical random ', ...
                         'fallback start is used.'];
                end
            end

        case 'user'
            [seedPoints, inputNorms] = local_prepare_seed_points( ...
                options.initialPoints, pointCount, m, 'initialPoints');
            [seedObjectiveF, ~, ~, objectiveInfo] = ...
                clifford_spherical_monogenic_objective( ...
                seedPoints, degree, alg);
            seed.source = 'user-supplied';
            seed.firstPointPolicy = 'user-supplied';
            seed.points = seedPoints;
            seed.initialPoints = seedPoints;
            seed.inputPointNorms = inputNorms;
            seed.initialObjectiveF = seedObjectiveF;
            seed.initialObjectiveInfo = local_compact_objective_info(objectiveInfo);
            seed.description = ...
                'Restart 1 is the normalized user-supplied configuration.';

        case 'continuation'
            [seedPoints, inputNorms] = local_prepare_seed_points( ...
                options.continuationPoints, pointCount, m, ...
                'continuationPoints');
            if options.continuationJitter > 0
                seedPoints = local_jitter_continuation( ...
                    seedPoints, options.continuationJitter, ...
                    options.anchorFirstPoint);
            end
            [seedObjectiveF, ~, ~, objectiveInfo] = ...
                clifford_spherical_monogenic_objective( ...
                seedPoints, degree, alg);
            seed.source = 'continuation';
            seed.firstPointPolicy = 'continuation';
            seed.points = seedPoints;
            seed.initialPoints = seedPoints;
            seed.inputPointNorms = inputNorms;
            seed.initialObjectiveF = seedObjectiveF;
            seed.initialObjectiveInfo = local_compact_objective_info(objectiveInfo);
            seed.description = ...
                ['Restart 1 is the normalized continuation configuration ', ...
                 '(optionally tangent-jittered).'];

        otherwise
            error('clifford_spherical_monogenic_design:InvalidSeedStrategy', ...
                'Unsupported seed strategy "%s".', options.seedStrategy);
    end
end

function [points, trace] = local_kernel_greedy_seed( ...
        m, degree, pointCount, alg, candidateCount)
% Select eta_1=e_1, then minimize the exact added F energy over candidates.

    points = zeros(pointCount, m);
    points(1, 1) = 1;
    trace = repmat(local_empty_greedy_trace(), pointCount, 1);
    trace(1).pointIndex = 1;
    trace(1).candidateCount = 0;
    trace(1).selectedCandidateIndex = NaN;
    trace(1).selectedIncrementalEnergy = 0;
    trace(1).minimumIncrementalEnergy = 0;
    trace(1).cumulativeObjectiveF = 0;
    trace(1).selectedPoint = points(1, :);

    currentObjectiveF = 0;
    for pointIndex = 2:pointCount
        candidates = local_random_sphere_points(candidateCount, m);
        candidateAddedEnergy = zeros(candidateCount, 1);
        prefix = points(1:(pointIndex - 1), :);
        for candidateIndex = 1:candidateCount
            [~, ~, ~, candidateInfo] = ...
                clifford_spherical_monogenic_objective( ...
                [prefix; candidates(candidateIndex, :)], degree, alg);
            candidateAddedEnergy(candidateIndex) = sum( ...
                candidateInfo.pairwiseCoefficientEnergies(end, 1:end-1));
        end
        [minimumEnergy, selectedCandidateIndex] = min(candidateAddedEnergy);
        points(pointIndex, :) = candidates(selectedCandidateIndex, :);
        currentObjectiveF = currentObjectiveF + minimumEnergy;

        trace(pointIndex).pointIndex = pointIndex;
        trace(pointIndex).candidateCount = candidateCount;
        trace(pointIndex).candidatePoints = candidates;
        trace(pointIndex).candidateEnergies = candidateAddedEnergy;
        trace(pointIndex).selectedCandidateIndex = selectedCandidateIndex;
        trace(pointIndex).selectedIncrementalEnergy = minimumEnergy;
        trace(pointIndex).minimumIncrementalEnergy = minimumEnergy;
        trace(pointIndex).cumulativeObjectiveF = currentObjectiveF;
        trace(pointIndex).selectedPoint = points(pointIndex, :);
    end

    % Re-evaluate once to remove any tiny accumulation error in the trace.
    finalObjectiveF = clifford_spherical_monogenic_objective( ...
        points, degree, alg);
    trace(end).cumulativeObjectiveF = finalObjectiveF;
end

function [points, trace] = local_flat_objective_seed(pointCount, m)
% A reproducible fallback only; no energy ordering exists in this case.

    points = local_random_sphere_points(pointCount, m);
    points(1, :) = 0;
    points(1, 1) = 1;
    trace = repmat(local_empty_greedy_trace(), pointCount, 1);
    for pointIndex = 1:pointCount
        trace(pointIndex).pointIndex = pointIndex;
        if pointIndex == 1
            trace(pointIndex).candidateCount = 0;
            trace(pointIndex).selectedCandidateIndex = NaN;
            trace(pointIndex).selectedIncrementalEnergy = 0;
            trace(pointIndex).minimumIncrementalEnergy = 0;
        else
            trace(pointIndex).candidateCount = 0;
            trace(pointIndex).selectedCandidateIndex = NaN;
            trace(pointIndex).selectedIncrementalEnergy = NaN;
            trace(pointIndex).minimumIncrementalEnergy = NaN;
        end
        trace(pointIndex).cumulativeObjectiveF = NaN;
        trace(pointIndex).selectedPoint = points(pointIndex, :);
    end
end

function points = local_jitter_continuation(points, jitter, anchorFirstPoint)
    tangentNoise = randn(size(points));
    tangentNoise = tangentNoise - ...
        bsxfun(@times, sum(tangentNoise .* points, 2), points);
    if anchorFirstPoint
        tangentNoise(1, :) = 0;
    end
    points = local_normalize_rows(points + jitter * tangentNoise);
end

function options = local_optimizer_options(options, seedPoints, seed)
    numRestarts = options.numRestarts;
    if strcmp(options.seedStrategy, 'kernel-greedy')
        if options.numRestartsWasExplicit && numRestarts ~= 1
            error('clifford_spherical_monogenic_design:KernelGreedyRestartCount', ...
                ['SEEDSTRATEGY=''kernel-greedy'' uses one start only. ', ...
                 'Use seedStrategy=''hybrid'' for a greedy start plus ', ...
                 'random restarts.']);
        end
        numRestarts = 1;
    end

    if isempty(options.randomSeed)
        optimizerRandomSeed = [];
    elseif seed.isKernelAware
        optimizerRandomSeed = local_next_random_seed(options.randomSeed);
    else
        optimizerRandomSeed = options.randomSeed;
    end

    options = rmfield(options, 'numRestartsWasExplicit');
    optimizerOptions = struct( ...
        'epsilon', options.epsilon, ...
        'relativeAmplificationLimit', options.relativeAmplificationLimit, ...
        'numRestarts', numRestarts, ...
        'maxIterations', options.maxIterations, ...
        'gradientTolerance', options.gradientTolerance, ...
        'initialStepSize', options.initialStepSize, ...
        'armijoCoefficient', options.armijoCoefficient, ...
        'backtrackingFactor', options.backtrackingFactor, ...
        'minimumStepSize', options.minimumStepSize, ...
        'feasibilityTolerance', options.feasibilityTolerance, ...
        'maximumPackedDimension', options.maximumPackedDimension, ...
        'initialPoints', seedPoints, ...
        'randomSeed', optimizerRandomSeed, ...
        'anchorFirstPoint', options.anchorFirstPoint, ...
        'verbose', options.verbose);
end

function design = local_assemble_design( ...
        m, degree, pointCount, diagonalLevel, alg, options, seed, ...
        optimizerOptions, optimizerResult)
    selected = optimizerResult.selected;

    design = struct();
    design.representation = 'spherical-monogenic-kernel-design';
    design.m = m;
    design.nBlades = alg.nBlades;
    design.packedDimension = pointCount * alg.nBlades;
    design.degree = degree;
    design.pointCount = pointCount;
    design.diagonalLevel = diagonalLevel;
    design.monogenicModuleRank = diagonalLevel;
    design.isSquareInterpolationEnsemble = pointCount == diagonalLevel;
    design.theoreticalPositiveDefinitenessPossible = ...
        pointCount <= diagonalLevel;
    design.seedStrategy = options.seedStrategy;
    design.seed = seed;
    design.seedPoints = seed.points;
    design.optionsUsed = local_public_options(options);
    design.optimizerOptionsUsed = optimizerOptions;
    design.optimizerResult = optimizerResult;
    design.summary = optimizerResult.summary;
    design.bestObjectiveRun = optimizerResult.bestObjectiveRun;
    design.bestStabilityRun = optimizerResult.bestStabilityRun;
    design.hasEpsilonTarget = optimizerResult.hasEpsilonTarget;
    design.epsilon = optimizerResult.epsilon;
    design.relativeAmplificationLimit = ...
        optimizerResult.relativeAmplificationLimit;
    design.feasibilityTolerance = optimizerResult.feasibilityTolerance;
    design.isEpsilonAchieved = optimizerResult.isEpsilonAchieved;
    design.isEpsilonAchievedWithinTolerance = ...
        optimizerResult.isEpsilonAchievedWithinTolerance;
    design.globalMinimumProved = false;
    design.selected = selected;
    design.hasSelectedConfiguration = ~isempty(selected);
    design.isUsable = ~isempty(selected);

    if isempty(selected)
        design.status = 'no-epsilon-feasible-candidate';
        design.centres = [];
        design.points = [];
        design.objectiveF = NaN;
        design.lambdaMin = NaN;
        design.lambdaMax = NaN;
        design.conditionNumber2 = NaN;
        design.relativeAmplification = NaN;
        design.isNumericallyPositiveDefinite = false;
        design.isUsableForInterpolation = false;
        design.isStableInterpolationCandidate = false;
        design.diagnostics = local_empty_diagnostics();
        design.gramEntries = [];
        design.kernelInfo = [];
        design.spectrum = [];
        design.leftRegularGramMatrix = [];
        return;
    end

    design.centres = selected.points;
    design.points = selected.points;
    design.objectiveF = selected.objectiveF;
    design.lambdaMin = selected.lambdaMin;
    design.lambdaMax = selected.lambdaMax;
    design.conditionNumber2 = selected.conditionNumber2;
    design.relativeAmplification = selected.relativeAmplification;
    design.isNumericallyPositiveDefinite = ...
        selected.isNumericallyPositiveDefinite;
    design.isUsableForInterpolation = selected.isNumericallyPositiveDefinite && ...
        (~optimizerResult.hasEpsilonTarget || ...
         optimizerResult.isEpsilonAchieved);
    design.isStableInterpolationCandidate = design.isUsableForInterpolation;
    design.gramEntries = selected.gramEntries;
    design.kernelInfo = selected.kernelInfo;
    design.spectrum = selected.spectrum;
    design.leftRegularGramMatrix = selected.spectrum.blockLeftRegularMatrix;
    design.diagnostics = local_selected_diagnostics(selected);
    if optimizerResult.hasEpsilonTarget && optimizerResult.epsilon == 0
        design.status = 'selected-psd-baseline';
    elseif optimizerResult.hasEpsilonTarget && optimizerResult.isEpsilonAchieved
        design.status = 'selected-strictly-feasible';
    elseif optimizerResult.hasEpsilonTarget
        design.status = 'selected-within-feasibility-tolerance';
    else
        design.status = 'selected-no-epsilon-target';
    end
end

function options = local_public_options(options)
    options = rmfield(options, 'numRestartsWasExplicit');
end

function seed = local_empty_seed(options, pointCount, m)
    seed = struct();
    seed.strategy = options.seedStrategy;
    seed.source = '';
    seed.description = '';
    seed.pointCount = pointCount;
    seed.ambientDimension = m;
    seed.requestedRandomSeed = options.randomSeed;
    seed.greedyRandomSeed = [];
    if any(strcmp(options.seedStrategy, {'hybrid', 'kernel-greedy'}))
        seed.greedyRandomSeed = options.randomSeed;
    end
    seed.optimizerRandomSeed = [];
    seed.optimizerNumRestarts = NaN;
    seed.randomRestartCount = NaN;
    seed.isKernelAware = false;
    seed.isFlatObjective = false;
    seed.firstPointPolicy = '';
    seed.canonicalAnchor = [];
    seed.candidateCount = options.greedyCandidateCount;
    seed.continuationJitter = options.continuationJitter;
    seed.points = [];
    seed.initialPoints = [];
    seed.inputPointNorms = [];
    seed.initialObjectiveF = NaN;
    seed.initialObjectiveInfo = [];
    seed.greedySkippedReason = '';
    seed.greedyTrace = repmat(local_empty_greedy_trace(), 0, 1);
end

function trace = local_empty_greedy_trace()
    trace = struct('pointIndex', NaN, 'candidateCount', NaN, ...
        'candidatePoints', [], 'candidateEnergies', [], ...
        'selectedCandidateIndex', NaN, 'selectedIncrementalEnergy', NaN, ...
        'minimumIncrementalEnergy', NaN, 'cumulativeObjectiveF', NaN, ...
        'selectedPoint', []);
end

function diagnostics = local_empty_diagnostics()
    diagnostics = struct('objectiveF', NaN, 'lambdaMin', NaN, ...
        'lambdaMax', NaN, 'conditionNumber2', NaN, ...
        'relativeAmplification', NaN, 'epsilonMargin', NaN, ...
        'iterations', NaN, 'exitFlag', '');
end

function diagnostics = local_selected_diagnostics(selected)
    diagnostics = struct('objectiveF', selected.objectiveF, ...
        'lambdaMin', selected.lambdaMin, 'lambdaMax', selected.lambdaMax, ...
        'conditionNumber2', selected.conditionNumber2, ...
        'relativeAmplification', selected.relativeAmplification, ...
        'epsilonMargin', selected.epsilonMargin, ...
        'iterations', selected.iterations, 'exitFlag', selected.exitFlag);
end

function compact = local_compact_objective_info(info)
    compact = struct();
    compact.diagonalLevel = info.diagonalLevel;
    compact.initialObjectiveF = info.objectiveF;
    compact.maxUsedUnitSphereResidual = info.maxUsedUnitSphereResidual;
    compact.isConstantOnSphere = info.isConstantOnSphere;
    compact.coefficientConvention = info.coefficientConvention;
end

function [points, inputNorms] = local_prepare_seed_points( ...
        points, pointCount, m, optionName)
    if isempty(points)
        error('clifford_spherical_monogenic_design:MissingSeedPoints', ...
            'OPTIONS.%s is required by the chosen seedStrategy.', optionName);
    end
    if ~(isnumeric(points) && isreal(points) && ismatrix(points) && ...
            size(points, 1) == pointCount && size(points, 2) == m && ...
            all(isfinite(points(:))))
        error('clifford_spherical_monogenic_design:InvalidSeedPoints', ...
            ['OPTIONS.%s must be a finite real POINTCOUNT-by-M matrix ', ...
             'with nonzero rows.'], optionName);
    end
    points = double(points);
    inputNorms = local_row_norms(points);
    if any(~isfinite(inputNorms)) || any(inputNorms == 0)
        error('clifford_spherical_monogenic_design:InvalidSeedPoints', ...
            'OPTIONS.%s must have only finite nonzero rows.', optionName);
    end
    points = bsxfun(@rdivide, points, inputNorms);
end

function points = local_random_sphere_points(pointCount, m)
    points = local_normalize_rows(randn(pointCount, m));
end

function points = local_normalize_rows(points)
    rowNorms = local_row_norms(points);
    if any(~isfinite(rowNorms)) || any(rowNorms == 0)
        error('clifford_spherical_monogenic_design:ZeroPoint', ...
            'A sphere point has a zero or nonfinite norm.');
    end
    points = bsxfun(@rdivide, points, rowNorms);
end

function rowNorms = local_row_norms(points)
    rowNorms = zeros(size(points, 1), 1);
    for row = 1:size(points, 1)
        rowNorms(row) = norm(points(row, :));
    end
end

function seed = local_next_random_seed(seed)
    seed = mod(seed + 1, 2^32);
end

function options = local_validate_options(options)
    if isempty(options)
        options = struct();
    end
    if ~(isstruct(options) && isscalar(options))
        error('clifford_spherical_monogenic_design:InvalidOptions', ...
            'OPTIONS must be a scalar structure.');
    end

    allowedFields = {'pointCount', 'seedStrategy', 'greedyCandidateCount', ...
        'initialPoints', 'continuationPoints', 'continuationJitter', ...
        'epsilon', 'relativeAmplificationLimit', 'numRestarts', ...
        'maxIterations', 'gradientTolerance', 'initialStepSize', ...
        'armijoCoefficient', 'backtrackingFactor', 'minimumStepSize', ...
        'feasibilityTolerance', 'maximumPackedDimension', 'randomSeed', ...
        'anchorFirstPoint', 'verbose'};
    suppliedFields = fieldnames(options);
    for fieldNumber = 1:numel(suppliedFields)
        if ~any(strcmp(suppliedFields{fieldNumber}, allowedFields))
            error('clifford_spherical_monogenic_design:UnknownOption', ...
                'OPTIONS contains the unsupported field "%s".', ...
                suppliedFields{fieldNumber});
        end
    end

    normalized = struct( ...
        'pointCount', [], ...
        'seedStrategy', 'hybrid', ...
        'greedyCandidateCount', 512, ...
        'initialPoints', [], ...
        'continuationPoints', [], ...
        'continuationJitter', 0, ...
        'epsilon', [], ...
        'relativeAmplificationLimit', [], ...
        'numRestarts', 8, ...
        'maxIterations', 800, ...
        'gradientTolerance', 1e-9, ...
        'initialStepSize', 1, ...
        'armijoCoefficient', 1e-4, ...
        'backtrackingFactor', 0.5, ...
        'minimumStepSize', 1e-14, ...
        'feasibilityTolerance', [], ...
        'maximumPackedDimension', 1024, ...
        'randomSeed', [], ...
        'anchorFirstPoint', true, ...
        'verbose', false, ...
        'numRestartsWasExplicit', any(strcmp(suppliedFields, 'numRestarts')));
    for fieldNumber = 1:numel(suppliedFields)
        fieldName = suppliedFields{fieldNumber};
        normalized.(fieldName) = options.(fieldName);
    end

    normalized.pointCount = local_validate_optional_positive_integer( ...
        normalized.pointCount, 'pointCount');
    normalized.seedStrategy = local_validate_seed_strategy( ...
        normalized.seedStrategy);
    normalized.greedyCandidateCount = local_validate_positive_integer( ...
        normalized.greedyCandidateCount, 'greedyCandidateCount');
    normalized.continuationJitter = local_validate_nonnegative_scalar( ...
        normalized.continuationJitter, 'continuationJitter');
    normalized.epsilon = local_validate_optional_nonnegative_scalar( ...
        normalized.epsilon, 'epsilon');
    normalized.relativeAmplificationLimit = ...
        local_validate_optional_amplification( ...
        normalized.relativeAmplificationLimit, 'relativeAmplificationLimit');
    if ~isempty(normalized.epsilon) && ...
            ~isempty(normalized.relativeAmplificationLimit)
        error('clifford_spherical_monogenic_design:AmbiguousEpsilon', ...
            ['Specify either OPTIONS.epsilon or ', ...
             'OPTIONS.relativeAmplificationLimit, not both.']);
    end
    normalized.numRestarts = local_validate_positive_integer( ...
        normalized.numRestarts, 'numRestarts');
    normalized.maxIterations = local_validate_nonnegative_integer( ...
        normalized.maxIterations, 'maxIterations', ...
        'clifford_spherical_monogenic_design:InvalidOption');
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
        error('clifford_spherical_monogenic_design:InvalidOption', ...
            'OPTIONS.minimumStepSize cannot exceed OPTIONS.initialStepSize.');
    end
    normalized.feasibilityTolerance = local_validate_optional_nonnegative_scalar( ...
        normalized.feasibilityTolerance, 'feasibilityTolerance');
    normalized.maximumPackedDimension = local_validate_packed_dimension( ...
        normalized.maximumPackedDimension, 'maximumPackedDimension');
    normalized.randomSeed = local_validate_optional_rng_seed( ...
        normalized.randomSeed, 'randomSeed');
    normalized.anchorFirstPoint = local_validate_boolean( ...
        normalized.anchorFirstPoint, 'anchorFirstPoint');
    normalized.verbose = local_validate_boolean(normalized.verbose, 'verbose');

    local_validate_seed_strategy_inputs(normalized);
    if strcmp(normalized.seedStrategy, 'kernel-greedy') && ...
            normalized.numRestartsWasExplicit && normalized.numRestarts ~= 1
        error('clifford_spherical_monogenic_design:KernelGreedyRestartCount', ...
            ['SEEDSTRATEGY=''kernel-greedy'' uses one start only. ', ...
             'Use seedStrategy=''hybrid'' for a greedy start plus ', ...
             'random restarts.']);
    end
    if normalized.continuationJitter > 0 && ...
            ~strcmp(normalized.seedStrategy, 'continuation')
        error('clifford_spherical_monogenic_design:UnusedContinuationJitter', ...
            ['OPTIONS.continuationJitter is only meaningful with ', ...
             'seedStrategy=''continuation''.']);
    end
    options = normalized;
end

function tf = local_has_positive_stability_target(options)
    tf = (~isempty(options.epsilon) && options.epsilon > 0) || ...
        ~isempty(options.relativeAmplificationLimit);
end

function local_validate_seed_strategy_inputs(options)
    hasInitial = ~isempty(options.initialPoints);
    hasContinuation = ~isempty(options.continuationPoints);
    switch options.seedStrategy
        case 'user'
            if ~hasInitial
                error('clifford_spherical_monogenic_design:MissingSeedPoints', ...
                    ['OPTIONS.initialPoints is required when ', ...
                     'seedStrategy=''user''.']);
            end
            if hasContinuation
                error('clifford_spherical_monogenic_design:AmbiguousSeedPoints', ...
                    ['Do not supply continuationPoints with ', ...
                     'seedStrategy=''user''.']);
            end
        case 'continuation'
            if ~hasContinuation
                error('clifford_spherical_monogenic_design:MissingSeedPoints', ...
                    ['OPTIONS.continuationPoints is required when ', ...
                     'seedStrategy=''continuation''.']);
            end
            if hasInitial
                error('clifford_spherical_monogenic_design:AmbiguousSeedPoints', ...
                    ['Do not supply initialPoints with ', ...
                     'seedStrategy=''continuation''.']);
            end
        otherwise
            if hasInitial || hasContinuation
                error('clifford_spherical_monogenic_design:UnusedSeedPoints', ...
                    ['initialPoints and continuationPoints are only accepted ', ...
                     'with seedStrategy=''user'' or ''continuation''.']);
            end
    end
end

function value = local_validate_seed_strategy(value)
    if isstring(value) && isscalar(value)
        value = char(value);
    end
    if ~(ischar(value) && size(value, 1) == 1)
        error('clifford_spherical_monogenic_design:InvalidSeedStrategy', ...
            'OPTIONS.seedStrategy must be one character vector or string scalar.');
    end
    value = lower(strtrim(value));
    allowed = {'hybrid', 'random', 'kernel-greedy', 'user', 'continuation'};
    if ~any(strcmp(value, allowed))
        error('clifford_spherical_monogenic_design:InvalidSeedStrategy', ...
            ['OPTIONS.seedStrategy must be ''hybrid'', ''random'', ', ...
             '''kernel-greedy'', ''user'', or ''continuation''.']);
    end
end

function m = local_validate_dimension(m)
    m = local_validate_nonnegative_integer(m, 'm', ...
        'clifford_spherical_monogenic_design:InvalidDimension');
    if m < 2 || m > 52
        error('clifford_spherical_monogenic_design:UnsupportedDimension', ...
            'M must be an integer between 2 and 52.');
    end
end

function value = local_validate_nonnegative_integer(value, fieldName, identifier)
    if ~(isnumeric(value) && isreal(value) && isscalar(value) && ...
            isfinite(value) && value == floor(value) && value >= 0)
        error(identifier, '%s must be one finite nonnegative integer.', ...
            upper(fieldName));
    end
    value = double(value);
end

function value = local_validate_positive_integer(value, fieldName)
    value = local_validate_nonnegative_integer(value, fieldName, ...
        'clifford_spherical_monogenic_design:InvalidOption');
    if value < 1
        error('clifford_spherical_monogenic_design:InvalidOption', ...
            'OPTIONS.%s must be a positive integer.', fieldName);
    end
end

function value = local_validate_optional_positive_integer(value, fieldName)
    if isempty(value)
        return;
    end
    value = local_validate_positive_integer(value, fieldName);
end

function value = local_validate_optional_rng_seed(value, fieldName)
    if isempty(value)
        return;
    end
    value = local_validate_nonnegative_integer(value, fieldName, ...
        'clifford_spherical_monogenic_design:InvalidOption');
    if value > 2^32 - 1
        error('clifford_spherical_monogenic_design:InvalidOption', ...
            ['OPTIONS.%s must be an integer between 0 and 2^32-1 for ', ...
             'the twister random-number generator.'], fieldName);
    end
end

function value = local_validate_optional_nonnegative_scalar(value, fieldName)
    if isempty(value)
        return;
    end
    value = local_validate_nonnegative_scalar(value, fieldName);
end

function value = local_validate_nonnegative_scalar(value, fieldName)
    if ~(isnumeric(value) && isreal(value) && isscalar(value) && ...
            isfinite(value) && value >= 0)
        error('clifford_spherical_monogenic_design:InvalidOption', ...
            'OPTIONS.%s must be one finite nonnegative scalar.', fieldName);
    end
    value = double(value);
end

function value = local_validate_optional_amplification(value, fieldName)
    if isempty(value)
        return;
    end
    if ~(isnumeric(value) && isreal(value) && isscalar(value) && ...
            isfinite(value) && value >= 1)
        error('clifford_spherical_monogenic_design:InvalidOption', ...
            ['OPTIONS.%s must be empty or one finite scalar at least 1. ', ...
             'A relative amplification bound below 1 is impossible.'], fieldName);
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

function value = local_validate_positive_scalar(value, fieldName)
    if ~(isnumeric(value) && isreal(value) && isscalar(value) && ...
            isfinite(value) && value > 0)
        error('clifford_spherical_monogenic_design:InvalidOption', ...
            'OPTIONS.%s must be one finite positive scalar.', fieldName);
    end
    value = double(value);
end

function value = local_validate_open_unit_scalar(value, fieldName)
    if ~(isnumeric(value) && isreal(value) && isscalar(value) && ...
            isfinite(value) && value > 0 && value < 1)
        error('clifford_spherical_monogenic_design:InvalidOption', ...
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
        error('clifford_spherical_monogenic_design:InvalidOption', ...
            'OPTIONS.%s must be one logical value or numeric 0/1.', fieldName);
    end
    value = logical(value);
end

function dimension = local_monogenic_dimension(m, degree)
    dimension = nchoosek(degree + m - 2, degree);
end
