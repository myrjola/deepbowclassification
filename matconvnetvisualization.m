% %% Download and compile MatConvNet
% untar(['http://www.vlfeat.org/matconvnet/download/matconvnet-1.0-beta23.tar.gz']);

if ~isa(@vl_simplenn, 'function_handle') % If matconvnet not initialized.
    cd matconvnet-1.0-beta23
    run matlab/vl_compilenn

    %% setup MatConvNet
    run matlab/vl_setupnn
    %
    cd ..
end

%% download a pre-trained CNN from the web if it has not been downloaded before
% cnnurl = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-f.mat'
cnnfile = 'imagenet-vgg-f.mat'

if ~exist(cnnfile, 'file')
    websave(cnnfile, cnnurl);
end

%% load the pre-trained CNN

net = load(cnnfile);

%% Get the weights
img = net.layers{1}.weights{1};
img = mat2gray(img);
% img = imresize(img, 5, 'nearest');
% sc(img, [0.0 1.0]);
% Uses imdisp from http://se.mathworks.com/matlabcentral/fileexchange/16233-sc-powerful-image-rendering
imdisp(img, 'Size', [8 8], 'Border', [0.01]);
