%RUN_MATRIX_EIGENSYSTEM_TEST Test the Clifford-matrix eigensystem utility.
%
% Open this file from the root of clifford-adjoint-matlab and press Run.

projectRoot = fileparts(mfilename('fullpath'));
addpath(fullfile(projectRoot, 'src'), ...
        fullfile(projectRoot, 'tests'), '-begin');

results = test_clifford_matrix_left_regular_eigensystem([4, 5]) %#ok<NOPTS>
