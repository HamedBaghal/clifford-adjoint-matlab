function [entries, info] = clifford_spherical_monogenic_gram( ...
        points, degree, alg, options)
%CLIFFORD_SPHERICAL_MONOGENIC_GRAM Build a spherical-monogenic Gram matrix.
%
%   [ENTRIES, INFO] = CLIFFORD_SPHERICAL_MONOGENIC_GRAM(POINTS, DEGREE, ALG)
%   constructs the Gram matrix of the degree-DEGREE spherical-monogenic
%   reproducing kernel at the rows of POINTS.  ALG must be
%   CLIFFORD_ALGEBRA(m) with m>=2, and POINTS must be an r-by-m real matrix
%   of unit-sphere points.
%
%   ENTRIES is an r-by-r-by-N coefficient tensor, N=ALG.nBlades, in the
%   canonical form used by CLIFFORD_MATRIX_LEFT_REGULAR_MATRIX:
%
%       ENTRIES(i,j,p) is the coefficient of blade mask p-1 in
%       K_degree(eta_i,eta_j).
%
%   For m>2, mu=(m-2)/2, and the kernel convention is
%
%       K_k(x,y) = ((k+m-2)/(m-2))*C_k^mu(<x,y>)
%                  + (x wedge y)*C_(k-1)^(mu+1)(<x,y>).
%
%   The m=2 branch uses its continuous Chebyshev limit, implemented by
%   CLIFFORD_SPHERICAL_MONOGENIC_KERNEL.  In every m>=2, the diagonal is the
%   scalar level binomial(k+m-2,k).  The lower triangle is formed by the
%   Clifford star, so ENTRIES is star-self-adjoint by construction.
%
%   INFO.objectiveF is the previous point-selection objective
%
%       sum_(i<j) |K_k(eta_i,eta_j)|_Cl^2,
%
%   where |.|_Cl is the Euclidean coefficient norm.  It is nonnegative and
%   is deliberately kept separate from a spectral stability criterion such
%   as lambda_min.  To obtain the latter, use the existing utility:
%
%       spectrum = clifford_matrix_left_regular_eigensystem(entries, alg);
%       lambdaMin = spectrum.hermitian.lambdaMin;
%
%   Keeping eigensolving outside this builder is important: optimization can
%   evaluate the kernel and objective cheaply, while a full spectral analysis
%   is requested only when needed.
%
%   OPTIONS is optional and must be a scalar structure with fields:
%
%       unitTolerance    allowed absolute error in each point norm; default
%                        100*eps*max(1,m).
%       normalizePoints  false (default) rejects nonunit rows; true divides
%                        each nonzero row by its Euclidean norm.

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
    nBlades = alg.nBlades;
    diagonalLevel = local_monogenic_dimension(alg.m, degree);
    if ~isfinite(diagonalLevel)
        error('clifford_spherical_monogenic_gram:DegreeTooLarge', ...
            ['DEGREE produces a nonfinite diagonal level in double ', ...
             'precision. Use a smaller degree or a scaled formulation.']);
    end
    entries = complex(zeros(pointCount, pointCount, nBlades));
    innerProducts = zeros(pointCount, pointCount);
    scalarKernelValues = zeros(pointCount, pointCount);
    bivectorKernelValues = zeros(pointCount, pointCount);
    pairEnergies = zeros(pointCount, pointCount);
    objectiveF = 0;
    kernelOptions = struct('unitTolerance', unitTolerance, ...
        'normalizePoints', false);

    for row = 1:pointCount
        entries(row, row, 1) = diagonalLevel;
        innerProducts(row, row) = 1;
        scalarKernelValues(row, row) = diagonalLevel;
        pairEnergies(row, row) = diagonalLevel^2;
        for column = (row + 1):pointCount
            [kernelCoefficients, kernelInfo] = ...
                clifford_spherical_monogenic_kernel( ...
                pointsUsed(row, :), pointsUsed(column, :), degree, alg, ...
                kernelOptions);
            entries(row, column, :) = reshape( ...
                kernelCoefficients, [1, 1, nBlades]);
            entries(column, row, :) = reshape( ...
                clifford_multivector_star(kernelCoefficients, alg), ...
                [1, 1, nBlades]);

            pairEnergy = sum(abs(kernelCoefficients).^2);
            objectiveF = objectiveF + pairEnergy;
            innerProducts(row, column) = kernelInfo.innerProduct;
            innerProducts(column, row) = kernelInfo.innerProduct;
            scalarKernelValues(row, column) = kernelInfo.scalarCoefficient;
            scalarKernelValues(column, row) = kernelInfo.scalarCoefficient;
            bivectorKernelValues(row, column) = kernelInfo.bivectorCoefficient;
            bivectorKernelValues(column, row) = kernelInfo.bivectorCoefficient;
            pairEnergies(row, column) = pairEnergy;
            pairEnergies(column, row) = pairEnergy;
        end
    end

    pointNormsAfter = sqrt(sum(pointsUsed.^2, 2));
    totalCoefficientFrobeniusEnergy = sum(abs(entries(:)).^2);
    offDiagonalCoefficientFrobeniusEnergy = 2 * objectiveF;

    info = struct();
    info.representation = 'spherical-monogenic-gram';
    info.m = alg.m;
    info.nBlades = nBlades;
    info.degree = degree;
    info.mu = (alg.m - 2) / 2;
    info.usesTwoDimensionalLimit = alg.m == 2;
    info.kernelFormula = local_kernel_formula_name(alg.m, degree);
    info.pointCount = pointCount;
    info.monogenicModuleRank = diagonalLevel;
    info.isSquareInterpolationEnsemble = pointCount == diagonalLevel;
    info.pointsInput = double(points);
    info.pointsUsed = pointsUsed;
    info.pointNormsBefore = pointNormsBefore;
    info.pointNormsAfter = pointNormsAfter;
    info.maxInputUnitSphereResidual = max(abs(pointNormsBefore - 1));
    info.maxUsedUnitSphereResidual = max(abs(pointNormsAfter - 1));
    info.unitTolerance = unitTolerance;
    info.normalizePoints = options.normalizePoints;
    info.innerProducts = innerProducts;
    info.scalarKernelValues = scalarKernelValues;
    info.bivectorKernelValues = bivectorKernelValues;
    info.pairwiseCoefficientEnergies = pairEnergies;
    info.diagonalLevel = diagonalLevel;
    info.objectiveF = objectiveF;
    info.offDiagonalPairEnergy = objectiveF;
    info.offDiagonalCoefficientFrobeniusEnergy = ...
        offDiagonalCoefficientFrobeniusEnergy;
    info.totalCoefficientFrobeniusEnergy = totalCoefficientFrobeniusEnergy;
    info.fullRegularOffDiagonalFrobeniusEnergy = ...
        2 * nBlades * objectiveF;
    info.fullRegularFrobeniusEnergy = ...
        nBlades * totalCoefficientFrobeniusEnergy;
    info.frobeniusScaling = sqrt(nBlades);
    info.coefficientTensorSize = [pointCount, pointCount, nBlades];
    info.coefficientConvention = ...
        'entries(i,j,p) is blade-mask p-1 coefficient of K_degree(eta_i,eta_j).';
    info.isStarSelfAdjointByConstruction = true;
end

function options = local_validate_options(options)
    if isempty(options)
        options = struct();
    end
    if ~(isstruct(options) && isscalar(options))
        error('clifford_spherical_monogenic_gram:InvalidOptions', ...
            'OPTIONS must be a scalar structure.');
    end

    allowedFields = {'unitTolerance', 'normalizePoints'};
    suppliedFields = fieldnames(options);
    for fieldNumber = 1:numel(suppliedFields)
        if ~any(strcmp(suppliedFields{fieldNumber}, allowedFields))
            error('clifford_spherical_monogenic_gram:UnknownOption', ...
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
        error('clifford_spherical_monogenic_gram:InvalidTolerance', ...
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
        error('clifford_spherical_monogenic_gram:InvalidBooleanOption', ...
            'OPTIONS.%s must be one logical value or numeric 0/1.', fieldName);
    end
    value = logical(value);
end

function local_validate_algebra(alg)
    isValid = isstruct(alg) && isscalar(alg) && isfield(alg, 'm') && ...
        isfield(alg, 'nBlades') && isfield(alg, 'maskToIndex') && ...
        isfield(alg, 'starBlade') && ...
        isa(alg.maskToIndex, 'function_handle') && ...
        isa(alg.starBlade, 'function_handle') && ...
        isnumeric(alg.m) && isreal(alg.m) && isscalar(alg.m) && ...
        isfinite(alg.m) && alg.m == floor(alg.m) && ...
        alg.m >= 0 && alg.m <= 52 && ...
        isnumeric(alg.nBlades) && isreal(alg.nBlades) && ...
        isscalar(alg.nBlades) && isfinite(alg.nBlades) && ...
        alg.nBlades == 2^alg.m;
    if ~isValid
        error('clifford_spherical_monogenic_gram:InvalidAlgebra', ...
            'ALG must be a structure returned by clifford_algebra.');
    end
    if alg.m < 2
        error('clifford_spherical_monogenic_gram:UnsupportedDimension', ...
            'Spherical-monogenic Gram construction requires ALG.m >= 2.');
    end
end

function degree = local_validate_degree(degree)
    if ~(isnumeric(degree) && isreal(degree) && isscalar(degree) && ...
            isfinite(degree) && degree == floor(degree) && degree >= 0)
        error('clifford_spherical_monogenic_gram:InvalidDegree', ...
            'DEGREE must be one finite nonnegative integer.');
    end
    degree = double(degree);
end

function [points, pointNorms] = local_prepare_points( ...
        points, m, unitTolerance, normalizePoints)
    if ~(isnumeric(points) && isreal(points) && ismatrix(points) && ...
            ~isempty(points) && size(points, 2) == m && ...
            all(isfinite(points(:))))
        error('clifford_spherical_monogenic_gram:InvalidPoints', ...
            ['POINTS must be a nonempty finite real r-by-ALG.m matrix, ', ...
             'with one point per row.']);
    end
    points = double(points);
    pointNorms = zeros(size(points, 1), 1);
    for row = 1:size(points, 1)
        pointNorms(row) = norm(points(row, :));
    end
    if any(pointNorms == 0)
        error('clifford_spherical_monogenic_gram:ZeroPoint', ...
            'Every row of POINTS must be nonzero.');
    end
    if normalizePoints
        points = bsxfun(@rdivide, points, pointNorms);
    elseif any(abs(pointNorms - 1) > unitTolerance)
        error('clifford_spherical_monogenic_gram:NonUnitPoint', ...
            ['Every row of POINTS must lie on the unit sphere within ', ...
             'OPTIONS.unitTolerance. Use normalizePoints=true to normalize ', ...
             'nonzero rows.']);
    end
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
