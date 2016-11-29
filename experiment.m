% %% Download and compile MatConvNet
% untar(['http://www.vlfeat.org/matconvnet/download/matconvnet-1.0-beta23.tar.gz']);
if ~exist('vl_simplenn') % If matconvnet not initialized.
     run matlab/vl_compilenn
    % setup MatConvNet
     run matlab/vl_setupnn
end

%% download a pre-trained CNN from the web if it has not been downloaded before
% cnnurl = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-f.mat'
% cnnfile = 'imagenet-vgg-f.mat'
% 
% if ~exist(cnnfile, 'file')
%     websave(cnnfile, cnnurl);
% end

%% load the pre-trained CNN


  net = load('imagenet-vgg-f.mat') ;
%  net = load('imagenet-vgg-verydeep-19.mat') ;
% net = load('imagenet-vgg-s.mat') ;

%% Apply classification on 10 categories

% Test dataset
outputFolder = 'caltech101'; % define output folder

if ~exist(outputFolder, 'dir') % download only once
    disp('Downloading 126MB Caltech101 data set...');
    % Download the compressed data set from the following location
    url = 'http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz';
    untar(url, outputFolder);
end

rootFolder = fullfile(outputFolder, '101_ObjectCategories');
categories = {'laptop', 'platypus', 'pizza', 'saxophone', 'soccer_ball', ...
              'accordion', 'ant', 'beaver', 'binocular', 'cannon'};

imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
imds.ReadFcn = @(filename) imread(filename);


%% Preprocess image and classify
res = zeros(1,size(imds.Files,1));
tic
for i=1:size(imds.Files,1)
    im = read(imds);
    if size(im, 3) == 1
        % greyscale image doesn't have BGR channels.
        im = repmat(im, [1 1 3]);
    end
    im_ = imresize(single(im), net.meta.normalization.imageSize(1:2));
    im_ = im_ - net.meta.normalization.averageImage;
    activations = vl_simplenn(net, im_);
    scores = squeeze(gather(activations(end).x));
    [~, res(i)] = max(scores);

%     Simple progress bar
     whos res
end
toc

classifications = arrayfun(@(i) net.meta.classes.description{i}, ...
                           res, ...
                           'UniformOutput', false);


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
mean(d(1:10))
%% Show classification result and visualization
for i=1:10
   
    [prob,idx] = sort(confmat(i,:),'descend');
    prob(1:4)
    s = sum(prob(1:4))-confmat(i,i)
    confusiongroups(idx(1:4))
end

figure;
imagesc(confmat(1:10,:))
colorbar('southoutside')
xlabel('Possible classifications');
ylabel('Test categories');
set(gca,'ytick',[1 2 3 4 5 6 7 8 9 10])
set(gca,'yticklabel',{'laptop','platypus','pizza','sax','soccer','accordion','ant','beaver','binocular','cannon'})
