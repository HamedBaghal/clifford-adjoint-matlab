%RUN_SPHERICAL_MONOGENIC_GRAM_TEST Test the spherical-monogenic Gram builder.
%
% Open this file from the root of clifford-adjoint-matlab and press Run.

projectRoot = fileparts(mfilename('fullpath'));
addpath(fullfile(projectRoot, 'src'), ...
        fullfile(projectRoot, 'tests'), '-begin');

results = test_clifford_spherical_monogenic_gram([2, 3, 4, 5]) %#ok<NOPTS>
