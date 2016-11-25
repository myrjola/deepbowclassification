%% Download and compile MatConvNet
untar(['http://www.vlfeat.org/matconvnet/download/matconvnet-1.0-beta23.tar.gz']);
cd matconvnet-1.0-beta23
run matlab/vl_compilenn


%% setup MatConvNet
run matlab/vl_setupnn

cd ..

%% download a pre-trained CNN from the web if it has not been downloaded before
cnnurl = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-f.mat'
cnnfile = 'imagenet-vgg-f.mat'

if ~exist(cnnfile, 'file')
    websave(cnnfile, cnnurl);
end

%% load the pre-trained CNN
net = load('imagenet-vgg-f.mat') ;

% We want to test the CNN with the provided test image. First we need to
% preprocess the image to fit it in the network, it needs to be resized and
% normalized by subtracting with the average image of all the images that the
% network has been trained with.

%% load and preprocess an image
imagefile = 'peppers.png';
im = imread(imagefile);
im_ = imresize(single(im), net.meta.normalization.imageSize(1:2));
im_ = im_ - net.meta.normalization.averageImage;

imshow(imread(imagefile))

%% runCN
res = vl_simplenn(net, im_);
scores = squeeze(gather(res(end).x));

%% we want the top ten scores and their indices
[toptenscores, toptenindices] = sort(scores, 'descend');
topten = arrayfun(@(i) net.meta.classes.description{i}, ...
                  toptenindices(1:10), ...
                  'UniformOutput', false);
table(toptenscores(1:10), topten);

% The example image are peppers, so at least in this case the CNN is accurate.
% Let's try a more involved example using the Caltech-101 dataset cite:caltech101.
% First we need to fetch the dataset. And create an image store of it. We have
% chosen five categories that the CNN is able to classify.

% Download the compressed data set from the following location
url = 'http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz';
% Store the output in a temporary folder
outputFolder = 'caltech101'; % define output folder

if ~exist(outputFolder, 'dir') % download only once
    disp('Downloading 126MB Caltech101 data set...');
    untar(url, outputFolder);
end

rootFolder = fullfile(outputFolder, '101_ObjectCategories');
categories = {'laptop', 'platypus', 'pizza', 'saxophone', 'soccer_ball'};

imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
imds.ReadFcn = @(filename) imread(filename);

% Now we send all the images to the CNN and extract the classification results.

%% Extract activations from the MatConvNet
res = [];
imds.reset;
while hasdata(imds)
    im = read(imds);
    if size(im, 3) == 1
        % greyscale image doesn't have BGR channels.
        im = repmat(im, [1 1 3]);
    end
    im_ = imresize(single(im), net.meta.normalization.imageSize(1:2));
    im_ = im_ - net.meta.normalization.averageImage;
    activations = vl_simplenn(net, im_);
    scores = squeeze(gather(activations(end).x));
    [maxscore, maxindex] = max(scores);
    res = vertcat(res, maxindex);

    % Simple progress bar
    whos res
end

classifications = arrayfun(@(i) net.meta.classes.description{i}, ...
                           res, ...
                           'UniformOutput', false);

classifications(1:5)

% The first images belong to the laptop category. So why does the third image get
% classified as a space bar?

imshow(imread(imds.Files{3}))

% Maybe the problem is that the laptop is in a box. One problem with CNN:s is that
% they are hard to troubleshoot because the features they detect can be very
% complex.

% Let's now measure the accuracy of the CNN for the chosen images. First we need
% to create a lookup table because Caltech101 uses different class names than the
% CNN we use.

categoriesCNN = {
    'laptop, laptop computer',
    'platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus',
    'pizza, pizza pie',
    'sax, saxophone',
    'soccer ball'
}

categoryMap = containers.Map(categories, categoriesCNN);

correctLabels = arrayfun(@(class) categoryMap(char(class)), cellstr(imds.Labels), ...
                         'UniformOutput', false);


% Now we are able to create a confusion matrix of the classification performance.

[confmat, confusiongroups] = confusionmat(correctLabels, classifications);
% Calculate the percentages
confmat = bsxfun(@rdivide, confmat, sum(confmat, 2));
% Get rid of the NaNs
confmat(isnan(confmat)) = 0;
d = diag(confmat);
% We are only concerned about the mean of the five classes we chose.
mean(d(1:5))

% Not too good results, let's see if we can figure out the problem with visualizations.


% A simple way to visualize a confusion matrix is to generate a grayscale image of the
% numerical data.

figure;
montage(imresize(confmat, 5, 'nearest'));

% What labels are the row and column indices corresponding to?

confusiongroups(1:10)
