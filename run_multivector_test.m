%RUN_MULTIVECTOR_TEST Test the general complex Clifford multivector product.
%
% Open this file from the root of clifford-adjoint-matlab and press Run.

projectRoot = fileparts(mfilename('fullpath'));
addpath(fullfile(projectRoot, 'src'), ...
        fullfile(projectRoot, 'tests'), '-begin');

results = test_clifford_multivector_product([4, 5]) %#ok<NOPTS>
