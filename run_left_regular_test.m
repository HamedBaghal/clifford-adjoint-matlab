%RUN_LEFT_REGULAR_TEST Test the faithful complex left-regular representation.
%
% Open this file from the root of clifford-adjoint-matlab and press Run.

projectRoot = fileparts(mfilename('fullpath'));
addpath(fullfile(projectRoot, 'src'), ...
        fullfile(projectRoot, 'tests'), '-begin');

results = test_clifford_left_regular_matrix([4, 5]) %#ok<NOPTS>
