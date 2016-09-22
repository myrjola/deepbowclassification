%% Add matcaffe to path
addpath('/Users/martin/git/caffe/matlab')

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

%% Test the net on one image
res = net.forward({im_data_repeated});


%% Get weights and output
weights = net.layers('fc8').params(1).get_data();
output = net.blobs('fc8').get_data();
