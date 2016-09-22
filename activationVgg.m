function res = activationVgg(net, trainingSet, featureLayer)
    res = [];
    ten_images = [];
    while hasdata(trainingSet)
        ten_images = cat(4, ten_images, read(trainingSet));

        if size(ten_images, 4) ~= 10
            continue;
        end

        net.forward({ten_images});
        features = net.blobs(featureLayer).get_data();
        res = [res, features];
        ten_images = [];
        whos res
    end
end
