%% Stolen from the DeepLearningImageClassificationExample
% http://se.mathworks.com/help/vision/examples/image-category-classification-using-deep-learning.html


function [trainingSet, testSet] = fetchCaltech101()
    % Download the compressed data set from the following location
    url = 'http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz';
    % Store the output in a temporary folder
    outputFolder = fullfile(tempdir, 'caltech101'); % define output folder

    if ~exist(outputFolder, 'dir') % download only once
        disp('Downloading 126MB Caltech101 data set...');
        untar(url, outputFolder);
    end

    rootFolder = fullfile(outputFolder, '101_ObjectCategories');
    categories = {'airplanes', 'ferry', 'laptop'};

    imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');

    tbl = countEachLabel(imds)

    minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category

    % Use splitEachLabel method to trim the set.
    % imds = splitEachLabel(imds, minSetCount, 'randomize');
    imds = splitEachLabel(imds, 10, 'randomize');

    % Notice that each set now has exactly the same number of images.
    countEachLabel(imds)

    imds.ReadFcn = @(filename)readAndPreprocessImageVGG(filename);

    [trainingSet, testSet] = splitEachLabel(imds, 0.3, 'randomize');
end