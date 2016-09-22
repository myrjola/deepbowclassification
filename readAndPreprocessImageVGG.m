function Iout = readAndPreprocessImageVGG(filename)

    im_data = caffe.io.load_image(filename);
    IM_WIDTH = 224;
    IM_HEIGHT = 224;
    im_data = imresize(im_data, [IM_WIDTH, IM_HEIGHT]);
    im_data = single(im_data);
    if size(im_data, 3) == 1
        % greyscale image doesn't have BGR channels.
        im_data = repmat(im_data, [1 1 3]);
    end
    Iout = im_data;
end
