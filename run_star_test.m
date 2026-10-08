%RUN_STAR_TEST Test the general complex Clifford adjoint.
%
% Open this file from the root of clifford-adjoint-matlab and press Run.

projectRoot = fileparts(mfilename('fullpath'));
addpath(fullfile(projectRoot, 'src'), ...
        fullfile(projectRoot, 'tests'), '-begin');

results = test_clifford_multivector_star([4, 5]) %#ok<NOPTS>
