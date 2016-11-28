%% Tests CNN on the given dataset
function confmat = testCNN(net, imds, categoryMap)
%% Extract activations from the MatConvNet
    res = [];
    imds.reset;
    tic
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
    toc

    classifications = arrayfun(@(i) net.meta.classes.description{i}, ...
                               res, ...
                               'UniformOutput', false);

    % Now we are able to create a confusion matrix of the classification performance.
    correctLabels = arrayfun(@(class) categoryMap(char(class)), cellstr(imds.Labels), ...
                             'UniformOutput', false);


    [confmat, confusiongroups] = confusionmat(correctLabels, classifications);
    % Calculate the percentages
    confmat = bsxfun(@rdivide, confmat, sum(confmat, 2));
    % Get rid of the NaNs
    confmat(isnan(confmat)) = 0;
end