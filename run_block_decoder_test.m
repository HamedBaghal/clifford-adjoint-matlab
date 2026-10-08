%RUN_BLOCK_DECODER_TEST Test the block left-regular decoder and validator.
%
% Open this file from the root of clifford-adjoint-matlab and press Run.

projectRoot = fileparts(mfilename('fullpath'));
addpath(fullfile(projectRoot, 'src'), ...
        fullfile(projectRoot, 'tests'), '-begin');

results = test_clifford_matrix_from_left_regular_matrix([4, 5]) %#ok<NOPTS>
