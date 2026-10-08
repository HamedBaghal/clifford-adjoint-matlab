%RUN_EIGENSYSTEM_TEST Test the regular-representation eigenvalue utility.
%
% Open this file from the root of clifford-adjoint-matlab and press Run.

projectRoot = fileparts(mfilename('fullpath'));
addpath(fullfile(projectRoot, 'src'), ...
        fullfile(projectRoot, 'tests'), '-begin');

results = test_clifford_left_regular_eigensystem([4, 5]) %#ok<NOPTS>
