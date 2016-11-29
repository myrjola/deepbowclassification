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
% 
% if ~exist(cnnfile, 'file')
%     websave(cnnfile, cnnurl);
% end

%% load the pre-trained CNN
% clear all

net = load(cnnfile) ;
% net = load('imagenet-vgg-verydeep-19.mat') ;
% net = load('imagenet-resnet-152-dag.mat');

% Download the compressed data set from the following location
url = 'http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz';
% Store the output in a temporary folder
outputFolder = 'caltech101'; % define output folder

if ~exist(outputFolder, 'dir') % download only once
    disp('Downloading 126MB Caltech101 data set...');
    untar(url, outputFolder);
end

rootFolder = fullfile(outputFolder, '101_ObjectCategories');
categories = {'laptop', 'platypus', 'pizza', 'saxophone', 'soccer_ball', ...
              'accordion', 'ant', 'beaver', 'binocular', 'cannon'};

% Let's now measure the accuracy of the CNN for the chosen images. First we need
% to create a lookup table because Caltech101 uses different class names than the
% CNN we use.

categoriesCNN = {
    'laptop, laptop computer',
    'platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus',
    'pizza, pizza pie',
    'sax, saxophone',
    'soccer ball',
    'accordion, piano accordion, squeeze box',
    'ant, emmet, pismire',
    'beaver',
    'binoculars, field glasses, opera glasses',
    'cannon'
}

categoryMap = containers.Map(categories, categoriesCNN);

imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
imds.ReadFcn = @(filename) imread(filename);

% Now we send all the images to the CNN and extract the classification results.

testCNN(net, imds, categoryMap);
% Not too good results, let's see if we can figure out the problem with visualizations.


% A simple way to visualize a confusion matrix is to generate a grayscale image of the
% numerical data.
%%
figure;
% A=imresize(confmat, 5, 'nearest');
% colormap jet
x = [1 size(confmat)];
% y = linspace(1,5,5);
imagesc(confmat(1:5,:))
colorbar('southoutside')
xlabel('Possible classifications');
ylabel('Test categories');
set(gca,'ytick',[1 2 3 4 5])
set(gca,'yticklabel',{'laptop','platypus','pizza','saxophone','soccer'})