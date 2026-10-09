function results = test_clifford_spherical_monogenic_optimizer()
%TEST_CLIFFORD_SPHERICAL_MONOGENIC_OPTIMIZER Test energy and sphere search.
%
%   RESULTS = TEST_CLIFFORD_SPHERICAL_MONOGENIC_OPTIMIZER() verifies the
%   pair-energy implementation, its Riemannian gradient, projected-gradient
%   descent, epsilon screening, and rank/constant-objective edge cases.

    savedRandomState = rng;
    restoreRandomState = onCleanup(@() rng(savedRandomState)); %#ok<NASGU>

    local_objective_and_gradient_checks();
    optimizerResult = local_known_two_centre_optimizer_check();
    local_constant_and_epsilon_checks();
    local_reproducibility_check();

    results = struct();
    results.m2ConstantObjectiveF = 3;
    results.m4DegreeOneObjectiveF = optimizerResult.selected.objectiveF;
    results.m4DegreeOneLambdaMin = optimizerResult.selected.lambdaMin;
    results.m4DegreeOneIterations = optimizerResult.selected.iterations;
    results.globalMinimumProved = optimizerResult.globalMinimumProved;
    fprintf(['Spherical-monogenic optimizer: objective, tangent gradient, ', ...
        'stability selection, and edge cases passed.\n']);
end

function local_objective_and_gradient_checks()
    alg = clifford_algebra(4);
    points = [1, 2, -1, 0; ...
             -2, 1, 1, 1; ...
              1, -1, 2, -2];
    points = local_normalize_rows(points);
    [F, euclideanGradient, tangentGradient, objectiveInfo] = ...
        clifford_spherical_monogenic_objective(points, 2, alg);
    [~, gramInfo] = clifford_spherical_monogenic_gram(points, 2, alg);

    local_assert_close(F, gramInfo.objectiveF);
    local_assert_close(objectiveInfo.pairwiseCoefficientEnergies, ...
        gramInfo.pairwiseCoefficientEnergies);
    assert(objectiveInfo.diagonalLevel == 6);
    assert(norm(sum(tangentGradient .* points, 2), inf) <= ...
        1e-11 * max(1, norm(tangentGradient, 'fro')));
    local_assert_close(tangentGradient, euclideanGradient - ...
        bsxfun(@times, sum(euclideanGradient .* points, 2), points));

    % A tolerance-accepted input is normalized internally to the exact sphere.
    nearSpherePoints = points;
    nearSpherePoints(1, :) = (1 + 20 * eps) * nearSpherePoints(1, :);
    [nearSphereF, ~, ~, nearSphereInfo] = ...
        clifford_spherical_monogenic_objective(nearSpherePoints, 2, alg);
    [~, normalizedGramInfo] = clifford_spherical_monogenic_gram( ...
        nearSpherePoints, 2, alg, struct('normalizePoints', true));
    local_assert_close(nearSphereF, F);
    local_assert_close(nearSphereF, normalizedGramInfo.objectiveF);
    assert(nearSphereInfo.maxUsedUnitSphereResidual <= 10 * eps);

    % F is O(m)-invariant, and the tangent gradient is equivariant.
    Q = [0, -1, 0, 0; 1, 0, 0, 0; 0, 0, 0, -1; 0, 0, 1, 0];
    [rotatedF, ~, rotatedGradient] = ...
        clifford_spherical_monogenic_objective(points * Q, 2, alg);
    local_assert_close(rotatedF, F);
    local_assert_close(rotatedGradient, tangentGradient * Q);

    % Central difference along product-sphere tangent directions.
    rng(916, 'twister');
    step = 2e-6;
    for directionNumber = 1:3
        direction = randn(size(points));
        direction = direction - ...
            bsxfun(@times, sum(direction .* points, 2), points);
        direction = direction / norm(direction, 'fro');
        plusPoints = local_normalize_rows(points + step * direction);
        minusPoints = local_normalize_rows(points - step * direction);
        plusF = clifford_spherical_monogenic_objective(plusPoints, 2, alg);
        minusF = clifford_spherical_monogenic_objective(minusPoints, 2, alg);
        finiteDifference = (plusF - minusF) / (2 * step);
        analyticDerivative = sum(sum(tangentGradient .* direction));
        local_assert_close_with_tolerance( ...
            finiteDifference, analyticDerivative, 5e-5);
    end
end

function result = local_known_two_centre_optimizer_check()
    % For m=4, k=1, r=2, F=1+8t^2 and lambda_min=3-sqrt(F).
    % The global F minimum is F=1 at t=0, lambda_min=2.
    initialPoints = [1, 0, 0, 0; 3/5, 4/5, 0, 0];
    options = struct( ...
        'epsilon', 1.9, ...
        'numRestarts', 1, ...
        'maxIterations', 1000, ...
        'gradientTolerance', 1e-11, ...
        'initialStepSize', 0.1, ...
        'initialPoints', initialPoints, ...
        'anchorFirstPoint', true);
    result = clifford_spherical_monogenic_optimize_centres(4, 1, 2, options);

    assert(result.theoreticalPositiveDefinitenessPossible);
    assert(result.hasEpsilonTarget);
    local_assert_close(result.epsilon, 1.9);
    assert(result.isEpsilonAchieved);
    assert(~isempty(result.selected));
    selected = result.selected;
    assert(selected.isFeasible);
    assert(selected.lambdaMin >= 1.9 - result.feasibilityTolerance);
    assert(abs(selected.points(1, :) * selected.points(2, :).') <= 1e-5);
    assert(selected.objectiveF <= 1 + 1e-8);
    local_assert_close_with_tolerance(selected.lambdaMin, ...
        3 - sqrt(selected.objectiveF), 1e-7);
    local_assert_close_with_tolerance(selected.lambdaMax, ...
        3 + sqrt(selected.objectiveF), 1e-7);
    assert(all(diff(selected.objectiveHistory) <= ...
        1e-11 * max(1, max(abs(selected.objectiveHistory)))));
    assert(selected.maxUnitSphereResidual <= 1e-12);
    assert(~result.globalMinimumProved);
    assert(~isempty(result.summary));
    assert(ismember('strictlyFeasible', result.summary.Properties.VariableNames));

    % A relative amplification cap gives epsilon=d/Amax^2.
    amplificationResult = clifford_spherical_monogenic_optimize_centres(4, 2, 1, ...
        struct('relativeAmplificationLimit', 2, 'numRestarts', 1, ...
        'maxIterations', 0, 'initialPoints', [1, 0, 0, 0]));
    local_assert_close(amplificationResult.epsilon, 1.5);
    assert(amplificationResult.isEpsilonAchieved);
    local_assert_close(amplificationResult.selected.lambdaMin, 6);
    local_assert_close(amplificationResult.selected.objectiveF, 0);

    exactDiagonalResult = clifford_spherical_monogenic_optimize_centres(4, 2, 1, ...
        struct('epsilon', 6, 'numRestarts', 1, 'maxIterations', 0, ...
        'initialPoints', [1, 0, 0, 0]));
    assert(exactDiagonalResult.isEpsilonAchieved);
    local_assert_close(exactDiagonalResult.selected.lambdaMin, 6);

    noFeasibleResult = clifford_spherical_monogenic_optimize_centres(4, 1, 2, ...
        struct('epsilon', 2.1, 'numRestarts', 1, 'maxIterations', 0, ...
        'initialPoints', [1, 0, 0, 0; 0, 1, 0, 0]));
    assert(~noFeasibleResult.isEpsilonAchieved);
    assert(isempty(noFeasibleResult.selected));
    local_assert_close(noFeasibleResult.bestStabilityRun.lambdaMin, 2);
end

function local_constant_and_epsilon_checks()
    % In m=2, F=choose(r,2) at every fixed degree and is exactly flat.
    alg2 = clifford_algebra(2);
    points2 = [1, 0; 0, 1; -1, 0];
    [F2, ~, tangentGradient2, objectiveInfo2] = ...
        clifford_spherical_monogenic_objective(points2, 3, alg2);
    local_assert_close(F2, 3);
    local_assert_close(tangentGradient2, zeros(size(points2)));
    assert(objectiveInfo2.isConstantOnSphere);

    result2 = clifford_spherical_monogenic_optimize_centres(2, 3, 3, ...
        struct('epsilon', 0, 'numRestarts', 1, 'maxIterations', 20, ...
        'initialPoints', points2));
    assert(strcmp(result2.runs(1).exitFlag, 'constant-objective'));
    local_assert_close(result2.selected.objectiveF, 3);
    assert(~result2.selected.isNumericallyPositiveDefinite);

    % Degree zero is also constant in every dimension.
    alg4 = clifford_algebra(4);
    points4 = [1, 0, 0, 0; 0, 1, 0, 0; 0, 0, 1, 0];
    [F0, ~, tangentGradient0] = ...
        clifford_spherical_monogenic_objective(points4, 0, alg4);
    local_assert_close(F0, 3);
    local_assert_close(tangentGradient0, zeros(size(points4)));

    % d_{2,k}=1: no positive spectral margin with more than one centre.
    local_assert_error(@() clifford_spherical_monogenic_optimize_centres( ...
        2, 3, 2, struct('epsilon', 1e-6)), ...
        'clifford_spherical_monogenic_optimize_centres:ImpossiblePositiveDefiniteTarget');
    local_assert_error(@() clifford_spherical_monogenic_optimize_centres( ...
        4, 2, 1, struct('epsilon', 6.1)), ...
        'clifford_spherical_monogenic_optimize_centres:EpsilonExceedsDiagonal');
    local_assert_error(@() clifford_spherical_monogenic_optimize_centres( ...
        4, 2, 1, struct('epsilon', 1, 'relativeAmplificationLimit', 2)), ...
        'clifford_spherical_monogenic_optimize_centres:AmbiguousEpsilon');
end

function local_reproducibility_check()
    options = struct('numRestarts', 2, 'maxIterations', 200, ...
        'gradientTolerance', 1e-10, 'initialStepSize', 0.1, ...
        'randomSeed', 71);
    first = clifford_spherical_monogenic_optimize_centres(4, 1, 2, options);
    second = clifford_spherical_monogenic_optimize_centres(4, 1, 2, options);
    local_assert_close(first.bestObjectiveRun.objectiveF, ...
        second.bestObjectiveRun.objectiveF);
    local_assert_close(first.bestObjectiveRun.points, ...
        second.bestObjectiveRun.points);
    assert(first.bestObjectiveRun.iterations == ...
        second.bestObjectiveRun.iterations);
    assert(strcmp(first.bestObjectiveRun.exitFlag, ...
        second.bestObjectiveRun.exitFlag));
end

function points = local_normalize_rows(points)
    points = bsxfun(@rdivide, points, sqrt(sum(points.^2, 2)));
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
