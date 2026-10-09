%RUN_SPHERICAL_MONOGENIC_DESIGN_TEST Test kernel-system design construction.
%
% Open this file from the root of clifford-adjoint-matlab and press Run.

projectRoot = fileparts(mfilename('fullpath'));
addpath(fullfile(projectRoot, 'src'), ...
        fullfile(projectRoot, 'tests'), '-begin');

results = test_clifford_spherical_monogenic_design() %#ok<NOPTS>
