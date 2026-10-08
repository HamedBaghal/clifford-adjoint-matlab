function results = test_clifford_algebra(dimensions)
%TEST_CLIFFORD_ALGEBRA Verify the general Cl_{0,m} blade implementation.
%
%   RESULTS = TEST_CLIFFORD_ALGEBRA() runs the test suite for m = 4 and m = 5.
%   RESULTS = TEST_CLIFFORD_ALGEBRA(DIMENSIONS) tests the requested vector of
%   dimensions.  Full associativity is checked over every triple of basis
%   blades, so this test is intentionally intended for small dimensions.

    if nargin == 0
        dimensions = [4, 5];
    end

    if ~(isnumeric(dimensions) && isreal(dimensions) && isvector(dimensions))
        error('test_clifford_algebra:InvalidDimensions', ...
            'DIMENSIONS must be a real numeric vector.');
    end

    template = struct('m', [], 'nBlades', [], 'associativityCases', []);
    results = repmat(template, 1, numel(dimensions));

    local_edge_case_checks();

    for testNumber = 1:numel(dimensions)
        m = dimensions(testNumber);
        alg = clifford_algebra(m);
        nBlades = alg.nBlades;

        assert(alg.m == m);
        assert(alg.nBlades == 2^m);
        assert(alg.scalarMask == uint64(0));
        assert(alg.fullMask == uint64(nBlades - 1));
        assert(strcmp(alg.label(uint64(0)), '1'));

        % Every blade has a valid mask/index round trip and a valid grade.
        for index = 1:nBlades
            mask = alg.indexToMask(index);
            assert(alg.maskToIndex(mask) == index);
            assert(alg.grade(mask) >= 0 && alg.grade(mask) <= m);
        end

        % The scalar blade is the multiplicative identity.
        for index = 1:nBlades
            mask = alg.indexToMask(index);
            [leftSign, leftProduct] = alg.multiplyMasks(uint64(0), mask);
            [rightSign, rightProduct] = alg.multiplyMasks(mask, uint64(0));
            assert(leftSign == 1 && leftProduct == mask);
            assert(rightSign == 1 && rightProduct == mask);
        end

        % Compare every blade product with an independent, sequential oracle.
        % This deliberately does not reuse the closed-form sign computation in
        % clifford_algebra.m.
        for leftIndex = 1:nBlades
            leftMask = alg.indexToMask(leftIndex);
            for rightIndex = 1:nBlades
                rightMask = alg.indexToMask(rightIndex);
                [signValue, productMask] = ...
                    alg.multiplyMasks(leftMask, rightMask);
                [oracleSign, oracleMask] = ...
                    local_sequential_product(leftMask, rightMask, m);
                [indexSign, productIndex] = ...
                    alg.multiplyIndices(leftIndex, rightIndex);

                assert(signValue == oracleSign);
                assert(productMask == oracleMask);
                assert(indexSign == signValue);
                assert(productIndex == alg.maskToIndex(productMask));

                % Involution identities on the canonical blade basis.
                [alphaLeft, alphaLeftMask] = alg.gradeInvolution(leftMask);
                [alphaRight, alphaRightMask] = alg.gradeInvolution(rightMask);
                [alphaProduct, alphaProductMask] = ...
                    alg.gradeInvolution(productMask);
                [alphaRhs, alphaRhsMask] = ...
                    alg.multiplyMasks(alphaLeftMask, alphaRightMask);
                assert(alphaProductMask == alphaRhsMask);
                assert(signValue * alphaProduct == ...
                    alphaLeft * alphaRight * alphaRhs);

                [reverseLeft, reverseLeftMask] = alg.reversion(leftMask);
                [reverseRight, reverseRightMask] = alg.reversion(rightMask);
                [reverseProduct, reverseProductMask] = ...
                    alg.reversion(productMask);
                [reverseRhs, reverseRhsMask] = ...
                    alg.multiplyMasks(reverseRightMask, reverseLeftMask);
                assert(reverseProductMask == reverseRhsMask);
                assert(signValue * reverseProduct == ...
                    reverseRight * reverseLeft * reverseRhs);

                [starLeft, starLeftMask] = alg.starBlade(leftMask);
                [starRight, starRightMask] = alg.starBlade(rightMask);
                [starProduct, starProductMask] = alg.starBlade(productMask);
                [starRhs, starRhsMask] = ...
                    alg.multiplyMasks(starRightMask, starLeftMask);
                assert(starProductMask == starRhsMask);
                assert(signValue * starProduct == ...
                    starRight * starLeft * starRhs);
            end
        end

        % Check e_i^2=-1 and e_i e_j=-e_j e_i for i~=j.
        for i = 1:m
            e_i = bitshift(uint64(1), i - 1);
            [squareSign, squareMask] = alg.multiplyMasks(e_i, e_i);
            assert(squareSign == -1 && squareMask == uint64(0));

            for j = i + 1:m
                e_j = bitshift(uint64(1), j - 1);
                [forwardSign, forwardMask] = alg.multiplyMasks(e_i, e_j);
                [reverseSign, reverseMask] = alg.multiplyMasks(e_j, e_i);
                expectedMask = bitxor(e_i, e_j);
                assert(forwardMask == expectedMask && forwardSign == 1);
                assert(reverseMask == expectedMask && reverseSign == -1);
            end
        end

        % Every canonical grade-r blade has square (-1)^(r(r+1)/2).
        for index = 1:nBlades
            mask = alg.indexToMask(index);
            r = alg.grade(mask);
            [squareSign, squareMask] = alg.multiplyMasks(mask, mask);
            assert(squareMask == uint64(0));
            assert(squareSign == (-1)^(r * (r + 1) / 2));
        end

        % Concrete low-dimensional checks make the sign convention explicit.
        if m >= 4
            e1 = uint64(1); e2 = uint64(2);
            e12 = uint64(3); e13 = uint64(5); e23 = uint64(6);
            [signValue, productMask] = alg.multiplyMasks(e1, e2);
            assert(signValue == 1 && productMask == e12);
            [signValue, productMask] = alg.multiplyMasks(e12, e23);
            assert(signValue == -1 && productMask == e13);
        end
        if m >= 5
            e5 = uint64(16); I5 = uint64(31); I4 = uint64(15);
            [signValue, productMask] = alg.multiplyMasks(e5, I5);
            assert(signValue == -1 && productMask == I4);
        end

        % Check reversion and Clifford conjugation on each blade.
        for index = 1:nBlades
            mask = alg.indexToMask(index);
            r = alg.grade(mask);
            [reverseSign, reverseMask] = alg.reversion(mask);
            [conjugateSign, conjugateMask] = alg.cliffordConjugation(mask);
            assert(reverseMask == mask);
            assert(conjugateMask == mask);
            assert(reverseSign == (-1)^(r * (r - 1) / 2));
            assert(conjugateSign == (-1)^(r * (r + 1) / 2));
        end

        % Associativity on the complete blade basis.
        for leftIndex = 1:nBlades
            leftMask = alg.indexToMask(leftIndex);
            for middleIndex = 1:nBlades
                middleMask = alg.indexToMask(middleIndex);
                [leftMiddleSign, leftMiddleMask] = ...
                    alg.multiplyMasks(leftMask, middleMask);

                for rightIndex = 1:nBlades
                    rightMask = alg.indexToMask(rightIndex);

                    [leftFinalSign, leftFinalMask] = ...
                        alg.multiplyMasks(leftMiddleMask, rightMask);
                    totalLeftSign = leftMiddleSign * leftFinalSign;

                    [middleRightSign, middleRightMask] = ...
                        alg.multiplyMasks(middleMask, rightMask);
                    [rightFinalSign, rightFinalMask] = ...
                        alg.multiplyMasks(leftMask, middleRightMask);
                    totalRightSign = middleRightSign * rightFinalSign;

                    assert(leftFinalMask == rightFinalMask);
                    assert(totalLeftSign == totalRightSign);
                end
            end
        end

        results(testNumber).m = m;
        results(testNumber).nBlades = nBlades;
        results(testNumber).associativityCases = nBlades^3;
        fprintf('Cl_{0,%d}: %d blades, %d associativity cases passed.\n', ...
            m, nBlades, nBlades^3);
    end
end

function [signValue, outputMask] = local_sequential_product(leftMask, rightMask, m)
% Independent product oracle: append the right generators one at a time.

    signValue = 1;
    outputMask = leftMask;

    for bit = 1:m
        if bitget(rightMask, bit)
            % Restore increasing order after appending e_bit on the right.
            higherCount = local_popcount(bitshift(outputMask, -bit));
            if mod(higherCount, 2) == 1
                signValue = -signValue;
            end

            if bitget(outputMask, bit)
                % The adjacent pair e_bit^2 contributes -1 and cancels.
                signValue = -signValue;
                outputMask = bitset(outputMask, bit, 0);
            else
                outputMask = bitset(outputMask, bit, 1);
            end
        end
    end
end

function count = local_popcount(mask)
    count = 0;
    while mask ~= uint64(0)
        mask = bitand(mask, mask - uint64(1));
        count = count + 1;
    end
end

function local_edge_case_checks()
% The algebra core also covers the scalar algebra Cl_{0,0} and Cl_{0,1}.

    alg0 = clifford_algebra(0);
    assert(alg0.nBlades == 1);
    [signValue, outputMask] = alg0.multiplyMasks(uint64(0), uint64(0));
    assert(signValue == 1 && outputMask == uint64(0));

    alg1 = clifford_algebra(1);
    [signValue, outputMask] = alg1.multiplyMasks(uint64(1), uint64(1));
    assert(signValue == -1 && outputMask == uint64(0));

    local_assert_error(@() clifford_algebra(-1));
    local_assert_error(@() clifford_algebra(1.5));
    local_assert_error(@() clifford_algebra(NaN));
    local_assert_error(@() alg1.multiplyMasks(uint64(2), uint64(0)));
    local_assert_error(@() alg1.indexToMask(3));
end

function local_assert_error(action)
    didError = false;
    try
        action();
    catch
        didError = true;
    end
    assert(didError);
end
