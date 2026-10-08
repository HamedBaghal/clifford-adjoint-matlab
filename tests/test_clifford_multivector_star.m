function results = test_clifford_multivector_star(dimensions)
%TEST_CLIFFORD_MULTIVECTOR_STAR Test the complex Clifford adjoint.
%
%   RESULTS = TEST_CLIFFORD_MULTIVECTOR_STAR() tests dimensions m=4 and
%   m=5.  Its reference answer computes the grade signs directly, without
%   using ALG.starBlade or any other involution supplied by ALG.

    if nargin == 0
        dimensions = [4, 5];
    end

    if ~(isnumeric(dimensions) && isreal(dimensions) && isvector(dimensions))
        error('test_clifford_multivector_star:InvalidDimensions', ...
            'DIMENSIONS must be a real numeric vector.');
    end

    local_edge_and_input_checks();
    savedRandomState = rng;
    restoreRandomState = onCleanup(@() rng(savedRandomState)); %#ok<NASGU>

    template = struct('m', [], 'nBlades', [], 'referenceResidual', [], ...
        'involutionResidual', [], 'antiAutomorphismResidual', [], ...
        'regularAdjointResidual', [], 'scalarNormResidual', []);
    results = repmat(template, 1, numel(dimensions));

    for testNumber = 1:numel(dimensions)
        m = dimensions(testNumber);
        alg = clifford_algebra(m);
        nBlades = alg.nBlades;

        % Deterministic complex data make failures reproducible.
        rng(2000 + m, 'twister');
        a = randn(nBlades, 1) + 1i * randn(nBlades, 1);
        b = randn(nBlades, 1) + 1i * randn(nBlades, 1);

        aStar = clifford_multivector_star(a, alg);
        reference = local_reference_star(a, m);
        referenceResidual = norm(aStar - reference, inf);
        local_assert_close(aStar, reference);
        assert(isequal(size(aStar), [nBlades, 1]));
        local_assert_close(clifford_multivector_star(a.', alg), aStar);

        % Every canonical blade is checked against the independent formula.
        local_canonical_blade_checks(alg, m);

        % STAR is conjugate-linear, rather than complex-linear.
        alpha = 2 - 3i;
        beta = -1 + 4i;
        antilinearLeft = clifford_multivector_star(alpha * a + beta * b, alg);
        antilinearRight = conj(alpha) * aStar + ...
            conj(beta) * clifford_multivector_star(b, alg);
        local_assert_close(antilinearLeft, antilinearRight);

        % STAR is an involution.
        doubleStar = clifford_multivector_star(aStar, alg);
        involutionResidual = norm(doubleStar - a, inf);
        local_assert_close(doubleStar, a);

        % STAR reverses geometric products: (a*b)^star = b^star*a^star.
        product = clifford_multivector_product(a, b, alg);
        starOfProduct = clifford_multivector_star(product, alg);
        reversedProduct = clifford_multivector_product( ...
            clifford_multivector_star(b, alg), aStar, alg);
        antiAutomorphismResidual = norm(starOfProduct - reversedProduct, inf);
        local_assert_close(starOfProduct, reversedProduct);
        local_basis_pair_checks(alg, m);

        % The regular representation has the usual matrix-adjoint property:
        % L_(a^star) = L_a^H.  The apostrophe is intentional in this check.
        leftRegularA = local_left_regular_matrix_from_product(a, alg);
        leftRegularStar = local_left_regular_matrix_from_product(aStar, alg);
        regularAdjointResidual = norm(leftRegularStar - leftRegularA', inf);
        local_assert_close(leftRegularStar, leftRegularA');

        % The scalar component of a^star*a is the positive coefficient norm.
        normProduct = clifford_multivector_product(aStar, a, alg);
        coefficientEnergy = sum(abs(a).^2);
        scalarNormResidual = abs(normProduct(1) - coefficientEnergy);
        local_assert_close(normProduct(1), coefficientEnergy);

        results(testNumber).m = m;
        results(testNumber).nBlades = nBlades;
        results(testNumber).referenceResidual = referenceResidual;
        results(testNumber).involutionResidual = involutionResidual;
        results(testNumber).antiAutomorphismResidual = antiAutomorphismResidual;
        results(testNumber).regularAdjointResidual = regularAdjointResidual;
        results(testNumber).scalarNormResidual = scalarNormResidual;
        fprintf(['Cl_{0,%d}: %d coefficient components; ', ...
            'complex Clifford star passed.\n'], m, nBlades);
    end
end

function local_canonical_blade_checks(alg, m)
    nBlades = alg.nBlades;
    for index = 1:nBlades
        blade = zeros(nBlades, 1);
        blade(index) = 1;
        expected = local_reference_star(blade, m);
        actual = clifford_multivector_star(blade, alg);
        assert(isequal(actual, expected));
    end
end

function local_basis_pair_checks(alg, m)
% Exhaustively check (e_I e_J)^star = e_J^star e_I^star.

    nBlades = alg.nBlades;
    for leftIndex = 1:nBlades
        leftBlade = zeros(nBlades, 1);
        leftBlade(leftIndex) = 1;
        for rightIndex = 1:nBlades
            rightBlade = zeros(nBlades, 1);
            rightBlade(rightIndex) = 1;
            actual = clifford_multivector_star( ...
                clifford_multivector_product(leftBlade, rightBlade, alg), alg);
            expected = clifford_multivector_product( ...
                clifford_multivector_star(rightBlade, alg), ...
                clifford_multivector_star(leftBlade, alg), alg);
            assert(isequal(actual, expected));
        end
    end

    if m >= 1
        e1 = zeros(nBlades, 1); e1(2) = 1;
        assert(isequal(clifford_multivector_star(e1, alg), -e1));
    end
    if m >= 2
        e12 = zeros(nBlades, 1); e12(4) = 1;
        assert(isequal(clifford_multivector_star(e12, alg), -e12));
    end
    if m >= 3
        e123 = zeros(nBlades, 1); e123(8) = 1;
        assert(isequal(clifford_multivector_star(e123, alg), e123));
    end
end

function aStar = local_reference_star(a, m)
% Independent grade-sign formula; it deliberately does not use ALG helpers.

    a = double(a(:));
    aStar = zeros(2^m, 1);
    for index = 1:numel(a)
        grade = local_popcount(uint64(index - 1));
        signValue = (-1)^(grade * (grade + 1) / 2);
        aStar(index) = signValue * conj(a(index));
    end
end

function leftRegular = local_left_regular_matrix_from_product(a, alg)
% Build L_a column by column with the independently tested product module.

    nBlades = alg.nBlades;
    leftRegular = zeros(nBlades, nBlades);
    for column = 1:nBlades
        basisBlade = zeros(nBlades, 1);
        basisBlade(column) = 1;
        leftRegular(:, column) = ...
            clifford_multivector_product(a, basisBlade, alg);
    end
end

function count = local_popcount(mask)
    count = 0;
    while mask ~= uint64(0)
        mask = bitand(mask, mask - uint64(1));
        count = count + 1;
    end
end

function local_assert_close(actual, expected)
    scale = max([1; abs(actual(:)); abs(expected(:))]);
    residual = norm(actual(:) - expected(:), inf);
    assert(residual <= 5000 * eps(scale));
end

function local_edge_and_input_checks()
    alg0 = clifford_algebra(0);
    scalar = 2 - 3i;
    assert(isequal(clifford_multivector_star(scalar, alg0), conj(scalar)));

    alg1 = clifford_algebra(1);
    one = [1; 0];
    e1 = [0; 1];
    assert(isequal(clifford_multivector_star(one, alg1), one));
    assert(isequal(clifford_multivector_star(e1, alg1), -e1));
    assert(isequal(clifford_multivector_star(1i * one, alg1), -1i * one));
    assert(isequal(clifford_multivector_star(1i * e1, alg1), 1i * e1));

    alg = clifford_algebra(2);
    local_assert_error(@() clifford_multivector_star([1; 2], alg), ...
        'clifford_multivector_star:InvalidCoefficientVector');
    local_assert_error(@() clifford_multivector_star(ones(4), alg));
    local_assert_error(@() clifford_multivector_star([1; 0; NaN; 0], alg));
    local_assert_error(@() clifford_multivector_star([1; 0; Inf; 0], alg));
    local_assert_error(@() clifford_multivector_star('abcd', alg));
    local_assert_error(@() clifford_multivector_star(zeros(4, 1), struct()));
end

function local_assert_error(action, expectedId)
    if nargin < 2
        expectedId = '';
    end
    didError = false;
    try
        action();
    catch exception
        didError = true;
        if ~isempty(expectedId)
            assert(strcmp(exception.identifier, expectedId));
        end
    end
    assert(didError);
end
