%RUN_FIRST_TEST Run the first arbitrary-dimension Clifford algebra check.
%
% Open this file from the root of clifford-adjoint-matlab and press Run.
% It tests the complete blade algebra in dimensions m=4 and m=5.

projectRoot = fileparts(mfilename('fullpath'));
addpath(fullfile(projectRoot, 'src'), ...
        fullfile(projectRoot, 'tests'), '-begin');

results = test_clifford_algebra([4, 5]) %#ok<NOPTS>
