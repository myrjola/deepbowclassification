%% Extract activations from the VGGNet
function res = activationVgg(net, X, featureLayer)
    res = [];
    ten_images = [];
    while hasdata(X)
        % We want to send ten images at the time to the VGGNet.
        ten_images = cat(4, ten_images, read(X));

        if size(ten_images, 4) ~= 10
            continue;
        end

        net.forward({ten_images});
        features = net.blobs(featureLayer).get_data();
        res = [res, features];
        ten_images = [];
        % Simple progress bar
        whos res
    end

    % Send the rest of the images.
    rest_of_the_images = ten_images;
    image_count = size(rest_of_the_images, 4);

    % size([], 4) returns 1
    if length(rest_of_the_images) && image_count > 0
        ten_images = padarray(rest_of_the_images, [0, 0, 0, 10-image_count], 'post');
        net.forward({ten_images});
        features = net.blobs(featureLayer).get_data();
        whos features;
        features = features(:,1:image_count);
        res = [res, features];
        whos res
    end
end