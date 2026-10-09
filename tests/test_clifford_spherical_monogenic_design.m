function results = test_clifford_spherical_monogenic_design()
%TEST_CLIFFORD_SPHERICAL_MONOGENIC_DESIGN Test seeded kernel-system builder.
%
%   RESULTS = TEST_CLIFFORD_SPHERICAL_MONOGENIC_DESIGN() checks default
%   dimensions, the auditable kernel-greedy trace, seed policies,
%   reproducibility, spectrum packaging, and strict epsilon semantics.

    savedRandomState = rng;
    restoreRandomState = onCleanup(@() rng(savedRandomState)); %#ok<NASGU>

    hybrid = local_hybrid_and_trace_checks();
    local_seed_policy_checks();
    oracle = local_target_and_failure_checks();
    local_higher_dimension_and_flat_checks();

    results = struct();
    results.defaultPointCount = hybrid.defaultPointCount;
    results.hybridSeedObjectiveF = hybrid.seedObjectiveF;
    results.m4DegreeOneObjectiveF = oracle.objectiveF;
    results.m4DegreeOneLambdaMin = oracle.lambdaMin;
    results.globalMinimumProved = oracle.globalMinimumProved;
    fprintf(['Spherical-monogenic design: defaults, seed policies, Gram ', ...
        'construction, and stability diagnostics passed.\n']);
end

function result = local_hybrid_and_trace_checks()
    % Default r is d_{4,2}=6; use a small candidate pool for test speed.
    defaultOptions = struct( ...
        'greedyCandidateCount', 11, ...
        'randomSeed', 401, ...
        'numRestarts', 1, ...
        'maxIterations', 0);
    defaultDesign = clifford_spherical_monogenic_design(4, 2, defaultOptions);
    assert(defaultDesign.pointCount == 6);
    assert(defaultDesign.diagonalLevel == 6);
    assert(defaultDesign.isSquareInterpolationEnsemble);
    assert(strcmp(defaultDesign.seedStrategy, 'hybrid'));
    assert(defaultDesign.seed.isKernelAware);
    local_assert_design_consistent(defaultDesign, 4, 2, 6);

    options = struct( ...
        'pointCount', 4, ...
        'seedStrategy', 'hybrid', ...
        'greedyCandidateCount', 23, ...
        'randomSeed', 402, ...
        'numRestarts', 2, ...
        'maxIterations', 0);
    first = clifford_spherical_monogenic_design(4, 2, options);
    second = clifford_spherical_monogenic_design(4, 2, options);
    local_assert_close(first.seed.points, second.seed.points);
    local_assert_close(first.centres, second.centres);
    local_assert_close(first.optimizerResult.runs(1).initialPoints, ...
        first.seed.points);
    assert(first.seed.optimizerRandomSeed == 403);
    assert(first.seed.randomRestartCount == 1);
    assert(first.seed.isKernelAware);
    assert(isequal(first.seed.canonicalAnchor, [1, 0, 0, 0]));
    local_assert_close(first.seed.points(1, :), [1, 0, 0, 0]);
    local_assert_unit_rows(first.seed.points);
    local_assert_greedy_trace(first, 4, 2);
    local_assert_design_consistent(first, 4, 2, 4);

    % The design wrapper must not change the caller's RNG state.
    rng(991, 'twister');
    expectedAfterCall = rand(1, 6);
    rng(991, 'twister');
    clifford_spherical_monogenic_design(4, 2, options); %#ok<NASGU>
    actualAfterCall = rand(1, 6);
    assert(isequal(expectedAfterCall, actualAfterCall));

    result = struct();
    result.defaultPointCount = defaultDesign.pointCount;
    result.seedObjectiveF = first.seed.initialObjectiveF;
end

function local_assert_greedy_trace(design, m, degree)
    alg = clifford_algebra(m);
    seedPoints = design.seed.points;
    trace = design.seed.greedyTrace;
    assert(numel(trace) == size(seedPoints, 1));
    for pointIndex = 2:numel(trace)
        stage = trace(pointIndex);
        assert(stage.pointIndex == pointIndex);
        assert(stage.candidateCount == size(stage.candidatePoints, 1));
        assert(stage.candidateCount == numel(stage.candidateEnergies));
        local_assert_unit_rows(stage.candidatePoints);

        prefix = seedPoints(1:(pointIndex - 1), :);
        recomputed = zeros(stage.candidateCount, 1);
        for candidateIndex = 1:stage.candidateCount
            [~, ~, ~, objectiveInfo] = ...
                clifford_spherical_monogenic_objective( ...
                [prefix; stage.candidatePoints(candidateIndex, :)], ...
                degree, alg);
            recomputed(candidateIndex) = sum( ...
                objectiveInfo.pairwiseCoefficientEnergies(end, 1:end-1));
        end
        local_assert_close(recomputed, stage.candidateEnergies);
        [minimumEnergy, firstIndex] = min(recomputed);
        assert(stage.selectedCandidateIndex == firstIndex);
        local_assert_close(stage.selectedIncrementalEnergy, minimumEnergy);
        local_assert_close(stage.minimumIncrementalEnergy, minimumEnergy);
        local_assert_close(stage.selectedPoint, ...
            stage.candidatePoints(firstIndex, :));
        local_assert_close(seedPoints(pointIndex, :), stage.selectedPoint);

        prefixF = clifford_spherical_monogenic_objective( ...
            seedPoints(1:pointIndex, :), degree, alg);
        local_assert_close(stage.cumulativeObjectiveF, prefixF);
    end
end

function local_seed_policy_checks()
    randomOptions = struct( ...
        'pointCount', 3, ...
        'seedStrategy', 'random', ...
        'randomSeed', 512, ...
        'numRestarts', 2, ...
        'maxIterations', 0);
    randomFirst = clifford_spherical_monogenic_design(4, 2, randomOptions);
    randomSecond = clifford_spherical_monogenic_design(4, 2, randomOptions);
    assert(isempty(randomFirst.seed.points));
    assert(isempty(randomFirst.optimizerOptionsUsed.initialPoints));
    assert(strcmp(randomFirst.seed.source, 'independent-random-restarts'));
    assert(randomFirst.seed.randomRestartCount == 2);
    local_assert_close(randomFirst.optimizerResult.runs(1).initialPoints, ...
        randomSecond.optimizerResult.runs(1).initialPoints);
    local_assert_unit_rows(randomFirst.optimizerResult.runs(1).initialPoints);

    userInput = [0, 2, 0, 0; 3, 0, 0, 0; 0, 0, -4, 0];
    userOptions = struct( ...
        'pointCount', 3, ...
        'seedStrategy', 'user', ...
        'initialPoints', userInput, ...
        'randomSeed', 513, ...
        'numRestarts', 1, ...
        'maxIterations', 0);
    userDesign = clifford_spherical_monogenic_design(4, 1, userOptions);
    expectedUser = [0, 1, 0, 0; 1, 0, 0, 0; 0, 0, -1, 0];
    local_assert_close(userDesign.seed.points, expectedUser);
    local_assert_close(userDesign.optimizerResult.runs(1).initialPoints, ...
        expectedUser);
    assert(strcmp(userDesign.seed.firstPointPolicy, 'user-supplied'));
    local_assert_design_consistent(userDesign, 4, 1, 3);

    continuationInput = [0, -5, 0, 0; 0, 0, 7, 0; 11, 0, 0, 0];
    continuationOptions = struct( ...
        'pointCount', 3, ...
        'seedStrategy', 'continuation', ...
        'continuationPoints', continuationInput, ...
        'continuationJitter', 0, ...
        'randomSeed', 514, ...
        'numRestarts', 1, ...
        'maxIterations', 0);
    continuationDesign = clifford_spherical_monogenic_design( ...
        4, 1, continuationOptions);
    expectedContinuation = [0, -1, 0, 0; 0, 0, 1, 0; 1, 0, 0, 0];
    local_assert_close(continuationDesign.seed.points, expectedContinuation);
    local_assert_close( ...
        continuationDesign.optimizerResult.runs(1).initialPoints, ...
        expectedContinuation);
    assert(strcmp(continuationDesign.seed.firstPointPolicy, 'continuation'));

    greedyOnly = clifford_spherical_monogenic_design(4, 2, struct( ...
        'pointCount', 3, ...
        'seedStrategy', 'kernel-greedy', ...
        'greedyCandidateCount', 13, ...
        'randomSeed', 515, ...
        'maxIterations', 0));
    assert(greedyOnly.optimizerOptionsUsed.numRestarts == 1);
    assert(greedyOnly.seed.randomRestartCount == 0);
    assert(strcmp(greedyOnly.seed.source, 'kernel-greedy'));

    local_assert_error(@() clifford_spherical_monogenic_design(4, 2, ...
        struct('pointCount', 3, 'seedStrategy', 'kernel-greedy', ...
        'numRestarts', 2)), ...
        'clifford_spherical_monogenic_design:KernelGreedyRestartCount');
    local_assert_error(@() clifford_spherical_monogenic_design(4, 2, ...
        struct('pointCount', 3, 'seedStrategy', 'user')), ...
        'clifford_spherical_monogenic_design:MissingSeedPoints');
    local_assert_error(@() clifford_spherical_monogenic_design(4, 2, ...
        struct('pointCount', 3, 'seedStrategy', 'hybrid', ...
        'initialPoints', eye(3, 4))), ...
        'clifford_spherical_monogenic_design:UnusedSeedPoints');
end

function result = local_target_and_failure_checks()
    initialPoints = [1, 0, 0, 0; 3/5, 4/5, 0, 0];
    success = clifford_spherical_monogenic_design(4, 1, struct( ...
        'pointCount', 2, ...
        'seedStrategy', 'user', ...
        'initialPoints', initialPoints, ...
        'randomSeed', 616, ...
        'epsilon', 1.9, ...
        'numRestarts', 1, ...
        'maxIterations', 1000, ...
        'gradientTolerance', 1e-11, ...
        'initialStepSize', 0.1));
    assert(success.isUsable);
    assert(success.isEpsilonAchieved);
    assert(success.isStableInterpolationCandidate);
    assert(strcmp(success.status, 'selected-strictly-feasible'));
    assert(success.objectiveF <= 1 + 1e-8);
    local_assert_close_with_tolerance(success.lambdaMin, 2, 1e-7);
    local_assert_design_consistent(success, 4, 1, 2);

    failure = clifford_spherical_monogenic_design(4, 1, struct( ...
        'pointCount', 2, ...
        'seedStrategy', 'user', ...
        'initialPoints', [1, 0, 0, 0; 0, 1, 0, 0], ...
        'epsilon', 2.1, ...
        'numRestarts', 1, ...
        'maxIterations', 0));
    assert(~failure.isUsable);
    assert(isempty(failure.selected));
    assert(isempty(failure.centres));
    assert(strcmp(failure.status, 'no-epsilon-feasible-candidate'));
    local_assert_close(failure.bestStabilityRun.lambdaMin, 2);

    local_assert_error(@() clifford_spherical_monogenic_design(4, 1, ...
        struct('pointCount', 4, 'seedStrategy', 'random', ...
        'epsilon', 1e-6, 'numRestarts', 1, 'maxIterations', 0)), ...
        'clifford_spherical_monogenic_design:ImpossiblePositiveDefiniteTarget');

    result = struct();
    result.objectiveF = success.objectiveF;
    result.lambdaMin = success.lambdaMin;
    result.globalMinimumProved = success.globalMinimumProved;
end

function local_higher_dimension_and_flat_checks()
    high = clifford_spherical_monogenic_design(5, 2, struct( ...
        'pointCount', 3, ...
        'seedStrategy', 'random', ...
        'randomSeed', 717, ...
        'numRestarts', 1, ...
        'maxIterations', 0));
    assert(high.diagonalLevel == 10);
    assert(high.nBlades == 32);
    assert(high.packedDimension == 96);
    local_assert_design_consistent(high, 5, 2, 3);

    flat = clifford_spherical_monogenic_design(2, 3, struct( ...
        'pointCount', 3, ...
        'seedStrategy', 'hybrid', ...
        'randomSeed', 718, ...
        'epsilon', 0, ...
        'numRestarts', 1, ...
        'maxIterations', 0));
    assert(flat.seed.isFlatObjective);
    assert(~flat.seed.isKernelAware);
    assert(~isempty(flat.seed.greedySkippedReason));
    local_assert_close(flat.objectiveF, 3);
    assert(strcmp(flat.status, 'selected-psd-baseline'));
    assert(~flat.isNumericallyPositiveDefinite);
    assert(~flat.isStableInterpolationCandidate);
end

function local_assert_design_consistent(design, m, degree, pointCount)
    assert(design.isUsable);
    assert(~isempty(design.selected));
    assert(isequal(size(design.centres), [pointCount, m]));
    local_assert_unit_rows(design.centres);
    local_assert_close(design.centres, design.selected.points);
    local_assert_close(design.centres, ...
        design.optimizerResult.selected.points);

    alg = clifford_algebra(m);
    [entries, info] = clifford_spherical_monogenic_gram( ...
        design.centres, degree, alg);
    local_assert_close(design.gramEntries, entries);
    local_assert_close(design.kernelInfo.objectiveF, info.objectiveF);
    local_assert_close(design.objectiveF, info.objectiveF);
    local_assert_close(design.spectrum.hermitian.lambdaMin, design.lambdaMin);
    local_assert_close(design.spectrum.hermitian.lambdaMax, design.lambdaMax);
    local_assert_close(design.diagnostics.objectiveF, design.objectiveF);
end

function local_assert_unit_rows(points)
    assert(~isempty(points));
    assert(max(abs(sum(points.^2, 2) - 1)) <= 1e-11);
end

function local_assert_close(actual, expected)
    assert(isequal(size(actual), size(expected)));
    scale = max([1; abs(actual(:)); abs(expected(:))]);
    residual = norm(actual(:) - expected(:), inf);
    dimensionFactor = max(1, sqrt(max(numel(actual), numel(expected))));
    assert(residual <= 5000 * dimensionFactor * eps(scale));
end

function local_assert_close_with_tolerance(actual, expected, relativeTolerance)
    assert(isequal(size(actual), size(expected)));
    scale = max([1; abs(actual(:)); abs(expected(:))]);
    residual = norm(actual(:) - expected(:), inf);
    assert(residual <= relativeTolerance * scale);
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
