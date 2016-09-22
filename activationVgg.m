function res = activationVgg(net, trainingSet, featureLayer)
  res = [];
  while hasdata(trainingSet)
    im_data = read(trainingSet);
    net.forward({im_data});
    features = net.blobs(featureLayer).get_data();
    % We are only concerned about the first column as the columns are all identical.
    features = features(:, 1);
    res = [res, features];
    whos res
  end
end
