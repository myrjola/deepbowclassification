%% Add matcaffe to path
addpath('/Users/martin/git/caffe/matlab')

%% Load images

[trainingSet, testSet] = fetchCaltech101();

%% Load the models
net_weights = ['VGG_ILSVRC_16_layers.caffemodel']
net_model = ['VGG_ILSVRC_16_layers_deploy.prototxt']
net = caffe.Net(net_model, net_weights, 'test')

%% Preprocess image
im_data = caffe.io.load_image('cat.jpg');
IM_WIDTH = 224;
IM_HEIGHT = 224;
im_data = imresize(im_data, [IM_WIDTH, IM_HEIGHT]);

%% So it seems that we need to repeat the image a couple of times
im_data_repeated = repmat(im_data, [1 1 1 9]);
im_data_repeated = cat(4, im_data_repeated, ones(224, 224, 3));
whos im_data;
whos im_data_repeated;

%% Test the net
trainingFeatures = activationVgg(net, trainingSet, 'fc8');

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
