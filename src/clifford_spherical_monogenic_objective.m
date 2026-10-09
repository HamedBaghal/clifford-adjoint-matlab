function [objectiveF, euclideanGradient, tangentGradient, info] = ...
        clifford_spherical_monogenic_objective(points, degree, alg, options)
%CLIFFORD_SPHERICAL_MONOGENIC_OBJECTIVE Energy and sphere gradient of F.
%
%   [F, EUCLIDEANGRADIENT, TANGENTGRADIENT, INFO] =
%   CLIFFORD_SPHERICAL_MONOGENIC_OBJECTIVE(POINTS, DEGREE, ALG) evaluates
%
%       F(POINTS) = sum_{i<j} |K_degree(eta_i,eta_j)|_Cl^2,
%
%   for the spherical-monogenic reproducing kernel implemented in
%   CLIFFORD_SPHERICAL_MONOGENIC_KERNEL.  POINTS is r-by-m, with one unit
%   sphere point per row, and ALG is CLIFFORD_ALGEBRA(m), m>=2.
%
%   TANGENTGRADIENT is the Riemannian gradient on the product sphere
%   (S^(m-1))^r.  It is the quantity used by
%   CLIFFORD_SPHERICAL_MONOGENIC_OPTIMIZE_CENTRES.  No Optimization Toolbox
%   is required.
%
%   OPTIONS is optional and may contain:
%
%       unitTolerance    allowed absolute error in a point norm; default
%                        100*eps*max(1,m).
%       normalizePoints  false (default) rejects rows outside the documented
%                        unit tolerance; true accepts any nonzero rows.
%
%   Every accepted row is normalized internally before the sphere formula and
%   tangent projection are evaluated.  This makes the returned objective an
%   exact function on the sphere even when an input differs from unit norm by
%   harmless floating-point roundoff.
%
%   For unit x,y and t=<x,y>, write K(x,y)=a(t)+(x wedge y)b(t).  Then
%
%       |K(x,y)|_Cl^2 = q(t) = a(t)^2 + (1-t^2)b(t)^2.
%
%   Therefore grad_{x_i} F = sum_{j~=i} q'(t_ij)x_j before tangent
%   projection.  This function evaluates q and q' directly, without
%   repeatedly constructing Clifford-valued kernel entries during an
%   optimization loop.

    if nargin < 4
        options = struct();
    end
    options = local_validate_options(options);
    local_validate_algebra(alg);
    degree = local_validate_degree(degree);

    if isempty(options.unitTolerance)
        unitTolerance = 100 * eps * max(1, alg.m);
    else
        unitTolerance = options.unitTolerance;
    end
    [pointsUsed, pointNormsBefore] = local_prepare_points( ...
        points, alg.m, unitTolerance, options.normalizePoints);

    pointCount = size(pointsUsed, 1);
    diagonalLevel = local_monogenic_dimension(alg.m, degree);
    if ~isfinite(diagonalLevel)
        error('clifford_spherical_monogenic_objective:DegreeTooLarge', ...
            ['DEGREE produces a nonfinite monogenic dimension in double ', ...
             'precision. Use a smaller degree or a scaled formulation.']);
    end

    objectiveF = 0;
    euclideanGradient = zeros(pointCount, alg.m);
    innerProducts = eye(pointCount);
    scalarKernelValues = diagonalLevel * eye(pointCount);
    bivectorKernelValues = zeros(pointCount);
    pairwiseCoefficientEnergies = diagonalLevel^2 * eye(pointCount);
    pairwiseEnergyDerivatives = zeros(pointCount);

    for row = 1:pointCount
        for column = (row + 1):pointCount
            rawInnerProduct = pointsUsed(row, :) * pointsUsed(column, :).';
            innerProduct = min(1, max(-1, rawInnerProduct));
            [pairEnergy, pairDerivative, scalarValue, bivectorValue] = ...
                local_pair_energy_and_derivative(innerProduct, degree, alg.m);
            if ~(isfinite(pairEnergy) && isfinite(pairDerivative) && ...
                    isfinite(scalarValue) && isfinite(bivectorValue))
                error('clifford_spherical_monogenic_objective:DegreeTooLarge', ...
                    ['DEGREE produces a nonfinite kernel-energy quantity in ', ...
                     'double precision. Use a smaller degree or a scaled ', ...
                     'formulation.']);
            end

            objectiveF = objectiveF + pairEnergy;
            euclideanGradient(row, :) = euclideanGradient(row, :) + ...
                pairDerivative * pointsUsed(column, :);
            euclideanGradient(column, :) = euclideanGradient(column, :) + ...
                pairDerivative * pointsUsed(row, :);

            innerProducts(row, column) = innerProduct;
            innerProducts(column, row) = innerProduct;
            scalarKernelValues(row, column) = scalarValue;
            scalarKernelValues(column, row) = scalarValue;
            bivectorKernelValues(row, column) = bivectorValue;
            bivectorKernelValues(column, row) = bivectorValue;
            pairwiseCoefficientEnergies(row, column) = pairEnergy;
            pairwiseCoefficientEnergies(column, row) = pairEnergy;
            pairwiseEnergyDerivatives(row, column) = pairDerivative;
            pairwiseEnergyDerivatives(column, row) = pairDerivative;
        end
    end

    tangentGradient = euclideanGradient - ...
        bsxfun(@times, sum(euclideanGradient .* pointsUsed, 2), pointsUsed);
    pointNormsAfter = sqrt(sum(pointsUsed.^2, 2));

    info = struct();
    info.representation = 'spherical-monogenic-pair-energy';
    info.m = alg.m;
    info.nBlades = alg.nBlades;
    info.degree = degree;
    info.mu = (alg.m - 2) / 2;
    info.usesTwoDimensionalLimit = alg.m == 2;
    info.pointCount = pointCount;
    info.monogenicModuleRank = diagonalLevel;
    info.diagonalLevel = diagonalLevel;
    info.pointsInput = double(points);
    info.pointsUsed = pointsUsed;
    info.pointNormsBefore = pointNormsBefore;
    info.pointNormsAfter = pointNormsAfter;
    info.maxInputUnitSphereResidual = max(abs(pointNormsBefore - 1));
    info.maxUsedUnitSphereResidual = max(abs(pointNormsAfter - 1));
    info.unitTolerance = unitTolerance;
    info.normalizePoints = options.normalizePoints;
    info.pointsWereNormalizedForSphere = ...
        any(abs(pointNormsBefore - 1) > 0);
    info.innerProducts = innerProducts;
    info.scalarKernelValues = scalarKernelValues;
    info.bivectorKernelValues = bivectorKernelValues;
    info.pairwiseCoefficientEnergies = pairwiseCoefficientEnergies;
    info.pairwiseEnergyDerivatives = pairwiseEnergyDerivatives;
    info.objectiveF = objectiveF;
    info.offDiagonalCoefficientFrobeniusEnergy = 2 * objectiveF;
    info.totalCoefficientFrobeniusEnergy = ...
        pointCount * diagonalLevel^2 + 2 * objectiveF;
    info.fullRegularOffDiagonalFrobeniusEnergy = ...
        2 * alg.nBlades * objectiveF;
    info.fullRegularFrobeniusEnergy = ...
        alg.nBlades * info.totalCoefficientFrobeniusEnergy;
    info.isConstantOnSphere = (alg.m == 2) || (degree == 0);
    info.euclideanGradientFrobeniusNorm = norm(euclideanGradient, 'fro');
    info.tangentGradientFrobeniusNorm = norm(tangentGradient, 'fro');
    info.coefficientConvention = ...
        'F uses the Euclidean blade-coefficient norm of each kernel entry.';
end

function options = local_validate_options(options)
    if isempty(options)
        options = struct();
    end
    if ~(isstruct(options) && isscalar(options))
        error('clifford_spherical_monogenic_objective:InvalidOptions', ...
            'OPTIONS must be a scalar structure.');
    end

    allowedFields = {'unitTolerance', 'normalizePoints'};
    suppliedFields = fieldnames(options);
    for fieldNumber = 1:numel(suppliedFields)
        if ~any(strcmp(suppliedFields{fieldNumber}, allowedFields))
            error('clifford_spherical_monogenic_objective:UnknownOption', ...
                'OPTIONS contains the unsupported field "%s".', ...
                suppliedFields{fieldNumber});
        end
    end

    normalized = struct('unitTolerance', [], 'normalizePoints', false);
    for fieldNumber = 1:numel(suppliedFields)
        fieldName = suppliedFields{fieldNumber};
        normalized.(fieldName) = options.(fieldName);
    end
    normalized.unitTolerance = local_validate_optional_tolerance( ...
        normalized.unitTolerance, 'unitTolerance');
    normalized.normalizePoints = local_validate_boolean( ...
        normalized.normalizePoints, 'normalizePoints');
    options = normalized;
end

function value = local_validate_optional_tolerance(value, fieldName)
    if isempty(value)
        return;
    end
    if ~(isnumeric(value) && isreal(value) && isscalar(value) && ...
            isfinite(value) && value >= 0)
        error('clifford_spherical_monogenic_objective:InvalidTolerance', ...
            'OPTIONS.%s must be empty or one finite nonnegative real scalar.', ...
            fieldName);
    end
    value = double(value);
end

function value = local_validate_boolean(value, fieldName)
    isLogical = islogical(value) && isscalar(value);
    isZeroOne = isnumeric(value) && isreal(value) && isscalar(value) && ...
        isfinite(value) && (value == 0 || value == 1);
    if ~(isLogical || isZeroOne)
        error('clifford_spherical_monogenic_objective:InvalidBooleanOption', ...
            'OPTIONS.%s must be one logical value or numeric 0/1.', fieldName);
    end
    value = logical(value);
end

function local_validate_algebra(alg)
    isValid = isstruct(alg) && isscalar(alg) && isfield(alg, 'm') && ...
        isfield(alg, 'nBlades') && ...
        isnumeric(alg.m) && isreal(alg.m) && isscalar(alg.m) && ...
        isfinite(alg.m) && alg.m == floor(alg.m) && ...
        alg.m >= 0 && alg.m <= 52 && ...
        isnumeric(alg.nBlades) && isreal(alg.nBlades) && ...
        isscalar(alg.nBlades) && isfinite(alg.nBlades) && ...
        alg.nBlades == 2^alg.m;
    if ~isValid
        error('clifford_spherical_monogenic_objective:InvalidAlgebra', ...
            'ALG must be a structure returned by clifford_algebra.');
    end
    if alg.m < 2
        error('clifford_spherical_monogenic_objective:UnsupportedDimension', ...
            'Spherical-monogenic energy requires ALG.m >= 2.');
    end
end

function degree = local_validate_degree(degree)
    if ~(isnumeric(degree) && isreal(degree) && isscalar(degree) && ...
            isfinite(degree) && degree == floor(degree) && degree >= 0)
        error('clifford_spherical_monogenic_objective:InvalidDegree', ...
            'DEGREE must be one finite nonnegative integer.');
    end
    degree = double(degree);
end

function [points, pointNorms] = local_prepare_points( ...
        points, m, unitTolerance, normalizePoints)
    if ~(isnumeric(points) && isreal(points) && ismatrix(points) && ...
            ~isempty(points) && size(points, 2) == m && ...
            all(isfinite(points(:))))
        error('clifford_spherical_monogenic_objective:InvalidPoints', ...
            ['POINTS must be a nonempty finite real r-by-ALG.m matrix, ', ...
             'with one point per row.']);
    end
    points = double(points);
    pointNorms = zeros(size(points, 1), 1);
    for row = 1:size(points, 1)
        pointNorms(row) = norm(points(row, :));
    end
    if any(pointNorms == 0)
        error('clifford_spherical_monogenic_objective:ZeroPoint', ...
            'Every row of POINTS must be nonzero.');
    end
    if ~normalizePoints && any(abs(pointNorms - 1) > unitTolerance)
        error('clifford_spherical_monogenic_objective:NonUnitPoint', ...
            ['Every row of POINTS must lie on the unit sphere within ', ...
             'OPTIONS.unitTolerance. Use normalizePoints=true to normalize ', ...
             'nonzero rows.']);
    end
    % The formulas below are intrinsic sphere formulas.  Normalize even a
    % tolerance-accepted row so its exact mathematical domain is respected.
    points = bsxfun(@rdivide, points, pointNorms);
end

function [energy, derivative, scalarValue, bivectorValue] = ...
        local_pair_energy_and_derivative(t, degree, m)
    if degree == 0
        scalarValue = 1;
        bivectorValue = 0;
        energy = 1;
        derivative = 0;
        return;
    end

    if m == 2
        scalarValue = local_chebyshev_t(degree, t);
        bivectorValue = local_chebyshev_u(degree - 1, t);
        % T_k(t)^2 + (1-t^2)U_{k-1}(t)^2 = 1 exactly on [-1,1].
        energy = 1;
        derivative = 0;
        return;
    end

    mu = (m - 2) / 2;
    scalarValue = ((degree + m - 2) / (m - 2)) * ...
        local_gegenbauer(degree, mu, t);
    bivectorValue = local_gegenbauer(degree - 1, m / 2, t);
    scalarDerivative = (degree + m - 2) * bivectorValue;
    if degree == 1
        bivectorDerivative = 0;
    else
        bivectorDerivative = m * local_gegenbauer( ...
            degree - 2, (m + 2) / 2, t);
    end

    energy = scalarValue^2 + (1 - t^2) * bivectorValue^2;
    derivative = 2 * scalarValue * scalarDerivative - ...
        2 * t * bivectorValue^2 + ...
        2 * (1 - t^2) * bivectorValue * bivectorDerivative;
end

function value = local_gegenbauer(degree, lambda, x)
    if degree == 0
        value = 1;
        return;
    end
    previous = 1;
    current = 2 * lambda * x;
    if degree == 1
        value = current;
        return;
    end
    for order = 2:degree
        next = (2 * (order + lambda - 1) * x * current - ...
            (order + 2 * lambda - 2) * previous) / order;
        previous = current;
        current = next;
    end
    value = current;
end

function value = local_chebyshev_t(degree, x)
    if degree == 0
        value = 1;
        return;
    end
    previous = 1;
    current = x;
    if degree == 1
        value = current;
        return;
    end
    for order = 2:degree %#ok<NASGU>
        next = 2 * x * current - previous;
        previous = current;
        current = next;
    end
    value = current;
end

function value = local_chebyshev_u(degree, x)
    if degree == 0
        value = 1;
        return;
    end
    previous = 1;
    current = 2 * x;
    if degree == 1
        value = current;
        return;
    end
    for order = 2:degree %#ok<NASGU>
        next = 2 * x * current - previous;
        previous = current;
        current = next;
    end
    value = current;
end

function dimension = local_monogenic_dimension(m, degree)
    dimension = nchoosek(degree + m - 2, degree);
end
