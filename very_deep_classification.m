%%
% This script uses the CNN from http://www.robots.ox.ac.uk/~vgg/research/very_deep/

%% Add matcaffe to path
addpath('../caffe/matlab')

%% Load images

[trainingSet, testSet] = fetchCaltech101();

%% Fetch the CNN

weightsURL = 'http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel';
weights = 'VGG_ILSVRC_16_layers.caffemodel';
modelURL = 'https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/0067c9b32f60362c74f4c445a080beed06b07eb3/VGG_ILSVRC_16_layers_deploy.prototxt';
model = 'VGG_ILSVRC_16_layers_deploy.prototxt';


% Fetch the model and weights if they don't already exist.
for file_url = {weights weightsURL; model modelURL}'
    file = char(file_url(1))
    url = char(file_url(2))
    if ~exist(file, 'file')
        websave(file, url);
    end
end

%% Load the CNN

% GPU is much faster, but can be unstable
% caffe.set_mode_gpu();

net = caffe.Net([model], [weights], 'test')

%% Test the net

% The feature layer is usually the layer right before the classification layer
featureLayer = 'fc7';

trainingFeatures = activationVgg(net, trainingSet, featureLayer);

% Get training labels from the trainingSet
trainingLabels = trainingSet.Labels;

% Train multiclass SVM classifier using a fast linear solver, and set
% 'ObservationsIn' to 'columns' to match the arrangement used for training
% features.
classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

testFeatures = activationVgg(net, testSet, 'fc8');

% Pass CNN image features to trained classifier, for some reason a transpose
% was needed.
predictedLabels = predict(classifier, testFeatures');

% Get the known labels
testLabels = testSet.Labels;

% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels, predictedLabels);

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2))

mean(diag(confMat))

% Visualize the confusion matrix
figure;
montage(imresize(confMat, 10));
