clear;
mode = 'GPU';
% add search path for caffe
addpath('/home/wang/work/caffe/matlab');
% add search path for mexopencv
addpath('/home/wang/disk_4T/wang_data/software/mexopencv');

input_root = '/home/wang/disk_4T/wang_data/original_dataset/CelebA/Img/img_celeba';
detect_root = '/home/wang/disk_4T/wang_data/original_dataset/CelebA/Img/detect_image';
align_root = '/home/wang/disk_4T/wang_data/original_dataset/CelebA/Img/align_image';

crop_size = [112, 96];
coord5points = [30.2946, 65.5318, 48.0252, 33.5493, 62.7299; ...
                51.6963, 51.5014, 71.7366, 92.3655, 92.2041];

if ~exist(input_root, 'dir')
    fprintf('Not found input_root -- > %s\n', input_root);
    return;
end

if exist(detect_root, 'dir')
    fprintf('Found detect_root --> %s, so we remove it\n', detect_root);
    rmdir(detect_root, 's');
end
mkdir(detect_root);

if exist(align_root, 'dir')
    fprintf('Found align_root --> %s, so we remove it\n', align_root);
    rmdir(align_root, 's');
end
mkdir(align_root)

%minimum size of face
minsize=20;

caffe_model_path='./model';

caffe.reset_all();
if strcmp(mode, 'GPU')
    caffe.set_mode_gpu()
    caffe.set_device(0)
elseif strcmp(mode, 'CPU')
    caffe.set_mode_cpu()
end

%three steps's threshold
threshold = [0.6 0.7 0.7];

%scale factor
factor = 0.709;

%load caffe models
prototxt_dir =strcat(caffe_model_path,'/det1.prototxt');
model_dir = strcat(caffe_model_path,'/det1.caffemodel');
PNet=caffe.Net(prototxt_dir,model_dir,'test');
prototxt_dir = strcat(caffe_model_path,'/det2.prototxt');
model_dir = strcat(caffe_model_path,'/det2.caffemodel');
RNet=caffe.Net(prototxt_dir,model_dir,'test');	
prototxt_dir = strcat(caffe_model_path,'/det3.prototxt');
model_dir = strcat(caffe_model_path,'/det3.caffemodel');
ONet=caffe.Net(prototxt_dir,model_dir,'test');
prototxt_dir =  strcat(caffe_model_path,'/det4.prototxt');
model_dir =  strcat(caffe_model_path,'/det4.caffemodel');
LNet = caffe.Net(prototxt_dir,model_dir,'test');
faces = cell(0);
img_list = dir(input_root);
img_list = {img_list.name};
for i=1:length(img_list)
    % i
    image_name = img_list{i};
    % image_name;
    % sprintf('image_name = %s', image_name)
    if strcmp(image_name, '.') || strcmp(image_name, '..') || ~endsWith(image_name, '.jpg')
        continue;
    end
    image_path = fullfile(input_root, image_name);
    fprintf('image_path = %s\n', image_path);
    
    if ~exist(image_path, 'file')
        fprintf('Not found file --> %s\n', image_path)
        return
    end
    img = cv.imread(image_path);
	%we recommend you to set minsize as x * short side
	%minl=min([size(img,1) size(img,2)]);
	%minsize=fix(minl*0.1)
    
    tic
    [boundingboxes, points] = detect_face(img, minsize, PNet, RNet, ONet, LNet, threshold, false, factor);
	toc

    faces{i, 1} = {boundingboxes};
	faces{i,2} = {points'};
	
	numbox = size(boundingboxes, 1);
    
    % show detection result
    img_detect = img;
    for j=1:numbox
        for id = 1:5
            key_point = [points(id, j), points(id+5,j)];
            img_detect = cv.drawMarker(img_detect, key_point, 'Color', [255, 0, 0], 'MarkerType', '*');
        end
        pt1 = boundingboxes(j, 1:2);
        pt2 = boundingboxes(j, 3:4);
        img_detect = cv.rectangle(img_detect, pt1, pt2, 'Color', [255, 0, 0], 'Thickness', 2);
        score = boundingboxes(j, 5);
        score = sprintf('%4f', score);
        img_detect = cv.putText(img_detect, score, pt1, 'Color', [255, 0, 0], 'Thickness', 2);
    end
    if ~exist(detect_root, 'dir')
        mkdir(detect_root)
    end
    
    cv.imwrite(fullfile(detect_root, image_name), img_detect);

    % show alignment result
    if numbox == 0
        continue;
    end
    [max_score, max_id] = max(boundingboxes(:, 5));
    if max_score < 0.9
        continue;
    end
    img_align = img;
    facial5points = [points(1:5, max_id)'; points(6:10, max_id)'];
    facial5points = double(facial5points);
    Tfm = cp2tform(facial5points', coord5points', 'similarity');
    img_align = imtransform(img_align, Tfm, 'XData', [1, crop_size(2)], ...
        'YData', [1, crop_size(1)], 'Size', crop_size);
    cv.imwrite(fullfile(align_root, image_name), img_align); 

end
