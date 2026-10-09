%RUN_SPHERICAL_MONOGENIC_OPTIMIZER_TEST Test projected sphere optimization.
%
% Open this file from the root of clifford-adjoint-matlab and press Run.

projectRoot = fileparts(mfilename('fullpath'));
addpath(fullfile(projectRoot, 'src'), ...
        fullfile(projectRoot, 'tests'), '-begin');

results = test_clifford_spherical_monogenic_optimizer() %#ok<NOPTS>
