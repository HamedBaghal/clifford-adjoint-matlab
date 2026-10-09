function [coefficients, info] = clifford_spherical_monogenic_kernel( ...
        x, y, degree, alg, options)
%CLIFFORD_SPHERICAL_MONOGENIC_KERNEL Reproducing kernel on a sphere.
%
%   [COEFFICIENTS, INFO] = CLIFFORD_SPHERICAL_MONOGENIC_KERNEL(X, Y,
%   DEGREE, ALG) evaluates the Clifford-valued spherical-monogenic kernel
%   K_degree(X,Y) for two points X,Y on S^(m-1), where m=ALG.m >= 2.
%   COEFFICIENTS is a column vector of length ALG.nBlades in mask order.
%
%   For m>2, with mu=(m-2)/2 and t=<X,Y>, the implemented convention is
%
%       K_k(X,Y) = ((k+m-2)/(m-2))*C_k^mu(t)
%                  + (X wedge Y)*C_(k-1)^(mu+1)(t).
%
%   The second term is zero when k=0.  The m=2 case is the continuous limit
%
%       K_0(X,Y) = 1,
%       K_k(X,Y) = T_k(t) + (X wedge Y)*U_(k-1)(t),  k>=1,
%
%   where T and U are Chebyshev polynomials.  In every m>=2,
%
%       K_k(X,X) = binomial(k+m-2,k).
%
%   OPTIONS is optional and must be a scalar structure with fields:
%
%       unitTolerance    allowed absolute error in norm(X)=norm(Y)=1;
%                        default 100*eps*max(1,m).
%       normalizePoints  false (default) rejects nonunit inputs; true divides
%                        each nonzero input vector by its Euclidean norm.
%
%   This kernel uses the Cl_{0,m} convention e_j^2=-1 and blade masks from
%   CLIFFORD_ALGEBRA.  The wedge coefficient for e_p*e_q, p<q, is
%   X(p)*Y(q)-X(q)*Y(p).

    if nargin < 5
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
    [x, xNormBefore] = local_prepare_point(x, alg.m, unitTolerance, ...
        options.normalizePoints, 'X');
    [y, yNormBefore] = local_prepare_point(y, alg.m, unitTolerance, ...
        options.normalizePoints, 'Y');

    rawInnerProduct = x * y.';
    innerProduct = min(1, max(-1, rawInnerProduct));
    scalarCoefficient = local_scalar_coefficient( ...
        degree, alg.m, innerProduct);
    if degree == 0
        bivectorCoefficient = 0;
    else
        bivectorCoefficient = local_gegenbauer( ...
            degree - 1, alg.m / 2, innerProduct);
    end
    if ~(isfinite(scalarCoefficient) && isfinite(bivectorCoefficient))
        error('clifford_spherical_monogenic_kernel:DegreeTooLarge', ...
            ['DEGREE produces a nonfinite kernel coefficient in double ', ...
             'precision. Use a smaller degree or a scaled formulation.']);
    end

    coefficients = complex(zeros(alg.nBlades, 1));
    coefficients(1) = scalarCoefficient;
    for firstCoordinate = 1:alg.m
        for secondCoordinate = (firstCoordinate + 1):alg.m
            wedgeCoefficient = x(firstCoordinate) * y(secondCoordinate) - ...
                x(secondCoordinate) * y(firstCoordinate);
            if wedgeCoefficient ~= 0 && bivectorCoefficient ~= 0
                bladeMask = bitset(bitset(uint64(0), firstCoordinate, 1), ...
                    secondCoordinate, 1);
                bladeIndex = alg.maskToIndex(bladeMask);
                coefficients(bladeIndex) = ...
                    bivectorCoefficient * wedgeCoefficient;
            end
        end
    end

    info = struct();
    info.representation = 'spherical-monogenic-reproducing-kernel';
    info.m = alg.m;
    info.nBlades = alg.nBlades;
    info.degree = degree;
    info.mu = (alg.m - 2) / 2;
    info.usesTwoDimensionalLimit = alg.m == 2;
    info.kernelFormula = local_kernel_formula_name(alg.m, degree);
    info.x = x;
    info.y = y;
    info.xNormBefore = xNormBefore;
    info.yNormBefore = yNormBefore;
    info.rawInnerProduct = rawInnerProduct;
    info.innerProduct = innerProduct;
    info.wasInnerProductClamped = rawInnerProduct ~= innerProduct;
    info.scalarCoefficient = scalarCoefficient;
    info.bivectorCoefficient = bivectorCoefficient;
    info.diagonalLevel = local_monogenic_dimension(alg.m, degree);
    info.coefficientConvention = ...
        'coefficients(k) is the blade-mask k-1 coefficient of K_degree(x,y).';
    info.unitTolerance = unitTolerance;
    info.normalizePoints = options.normalizePoints;
end

function options = local_validate_options(options)
    if isempty(options)
        options = struct();
    end
    if ~(isstruct(options) && isscalar(options))
        error('clifford_spherical_monogenic_kernel:InvalidOptions', ...
            'OPTIONS must be a scalar structure.');
    end

    allowedFields = {'unitTolerance', 'normalizePoints'};
    suppliedFields = fieldnames(options);
    for fieldNumber = 1:numel(suppliedFields)
        if ~any(strcmp(suppliedFields{fieldNumber}, allowedFields))
            error('clifford_spherical_monogenic_kernel:UnknownOption', ...
                'OPTIONS contains the unsupported field "%s".', ...
                suppliedFields{fieldNumber});
        end
    end

    normalized = struct();
    normalized.unitTolerance = [];
    normalized.normalizePoints = false;
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
        error('clifford_spherical_monogenic_kernel:InvalidTolerance', ...
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
        error('clifford_spherical_monogenic_kernel:InvalidBooleanOption', ...
            'OPTIONS.%s must be one logical value or numeric 0/1.', fieldName);
    end
    value = logical(value);
end

function local_validate_algebra(alg)
    isValid = isstruct(alg) && isscalar(alg) && isfield(alg, 'm') && ...
        isfield(alg, 'nBlades') && isfield(alg, 'maskToIndex') && ...
        isa(alg.maskToIndex, 'function_handle') && ...
        isnumeric(alg.m) && isreal(alg.m) && isscalar(alg.m) && ...
        isfinite(alg.m) && alg.m == floor(alg.m) && ...
        alg.m >= 0 && alg.m <= 52 && ...
        isnumeric(alg.nBlades) && isreal(alg.nBlades) && ...
        isscalar(alg.nBlades) && isfinite(alg.nBlades) && ...
        alg.nBlades == 2^alg.m;
    if ~isValid
        error('clifford_spherical_monogenic_kernel:InvalidAlgebra', ...
            ['ALG must be a structure returned by clifford_algebra, with ', ...
             'a valid maskToIndex function handle.']);
    end
    if alg.m < 2
        error('clifford_spherical_monogenic_kernel:UnsupportedDimension', ...
            'Spherical-monogenic kernels in this utility require ALG.m >= 2.');
    end
end

function degree = local_validate_degree(degree)
    if ~(isnumeric(degree) && isreal(degree) && isscalar(degree) && ...
            isfinite(degree) && degree == floor(degree) && degree >= 0)
        error('clifford_spherical_monogenic_kernel:InvalidDegree', ...
            'DEGREE must be one finite nonnegative integer.');
    end
    degree = double(degree);
end

function [point, normBefore] = local_prepare_point( ...
        point, m, unitTolerance, normalizePoint, pointName)
    if ~(isnumeric(point) && isreal(point) && isvector(point) && ...
            numel(point) == m && all(isfinite(point(:))))
        error('clifford_spherical_monogenic_kernel:InvalidPoint', ...
            '%s must be one finite real vector with exactly ALG.m entries.', ...
            pointName);
    end
    point = double(point(:)).';
    normBefore = norm(point);
    if normBefore == 0
        error('clifford_spherical_monogenic_kernel:ZeroPoint', ...
            '%s must be nonzero.', pointName);
    end
    if normalizePoint
        point = point / normBefore;
    elseif abs(normBefore - 1) > unitTolerance
        error('clifford_spherical_monogenic_kernel:NonUnitPoint', ...
            ['%s must lie on the unit sphere within OPTIONS.unitTolerance. ', ...
             'Use normalizePoints=true to normalize nonzero inputs.'], pointName);
    end
end

function value = local_scalar_coefficient(degree, m, innerProduct)
    if m == 2
        value = local_chebyshev_t(degree, innerProduct);
        return;
    end
    mu = (m - 2) / 2;
    value = ((degree + m - 2) / (m - 2)) * ...
        local_gegenbauer(degree, mu, innerProduct);
end

function value = local_gegenbauer(degree, lambda, x)
% Three-term recurrence, valid here for lambda>0 and integer degree>=0.

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

function dimension = local_monogenic_dimension(m, degree)
    dimension = nchoosek(degree + m - 2, degree);
end

function name = local_kernel_formula_name(m, degree)
    if m == 2 && degree > 0
        name = 'two-dimensional Chebyshev limit';
    elseif m == 2
        name = 'constant degree-zero kernel';
    else
        name = 'Gegenbauer spherical-monogenic kernel';
    end
end
