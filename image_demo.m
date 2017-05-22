clear;
mode = 'GPU';
show_detect = false;

% add search path for caffe
addpath('/home/wang/work/caffe/matlab');
% add search path for mexopencv
addpath('/home/wang/disk_4T/wang_data/software/mexopencv');

data_root = '/home/wang/disk_4T/wang_data/original_dataset/EthnicFace/temp';
input_root = fullfile(data_root, 'spider_image');
detect_root = fullfile(data_root, 'detect_image');
crop_root = fullfile(data_root, 'crop_image')
align_root = fullfile(data_root, 'align_image');

% input_label_txt = fullfile(data_root, '1.txt');
% output_label_txt = fullfile(data_root, 'output_label.txt');
% no_face_txt = fullfile(data_root, 'no_face.txt');

% output_label_fid = fopen(output_label_txt, 'w');
% no_face_fid = fopen(no_face_txt, 'w');

% crop_size = [112, 96];
% coord5points = [30.2946, 65.5318, 48.0252, 33.5493, 62.7299; ...
%                 51.6963, 51.5014, 71.7366, 92.3655, 92.2041];

base_crop_size = [218, 178];
base_coord5points = [70.0, 108.0, 85.0, 73.0, 104.0; ...
                113.0, 111.0, 136.0, 154.0, 153.0];

crop_size = [300, 200];
diff_size = crop_size - base_crop_size;
diff_size(1) = diff_size(1) * 0.3;
diff_size(2) = diff_size(2) * 0.5;
coord5points = base_coord5points + diff_size';

if ~exist(input_root, 'dir')
    fprintf('Not found input_root -- > %s\n', input_root);
    return;
end

if exist(detect_root, 'dir')
    fprintf('Found detect_root --> %s, so we remove it\n', detect_root);
    rmdir(detect_root, 's');
end
mkdir(detect_root);

if exist(crop_root, 'dir')
    fprintf('Found crop_root --> %s, so we remove it\n', crop_root);
    rmdir(crop_root, 's');
end
mkdir(crop_root);

if exist(align_root, 'dir')
    fprintf('Found align_root --> %s, so we remove it\n', align_root);
    rmdir(align_root, 's');
end
mkdir(align_root);

%minimum size of face
minsize=20;

caffe_model_path='./model';

caffe.reset_all();
if strcmp(mode, 'GPU')
    caffe.set_mode_gpu();
    caffe.set_device(0);
elseif strcmp(mode, 'CPU')
    caffe.set_mode_cpu();
end

%three steps's threshold
threshold = [0.6, 0.7, 0.7];

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

% faces = cell(0);
img_list = dir(input_root);
img_list = {img_list.name};
% img_list = input_label_data.rowheaders;

for i = 1:length(img_list)
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
        fprintf('Not found file --> %s\n', image_path);
        return;
    end
    img = 0;
    try
      img = cv.imread(image_path);
    catch
      warning('Fail to read image_path --> %s\n', image_path);
      continue;
    end
    % img = imread(image_path);
	  
    %we recommend you to set minsize as x * short side
	  %minl=min([size(img,1) size(img,2)]);
	  %minsize=fix(minl*0.1)
    
    tic
    [boundingboxes, points] = detect_face(img, minsize, PNet, RNet, ONet, LNet, threshold, false, factor);
	  toc
	
	  numbox = size(boundingboxes, 1);
     
    if numbox == 0
      fprintf('numbox == 0 --> %s\n', image_name);
      continue;
    end

    [max_score, max_id] = max(boundingboxes(:, 5)); 
    
    % for detect image
    img_detect = img;
    for id = 1:5
      key_point = [points(id, max_id), points(id+5, max_id)];
      img_detect = cv.drawMarker(img_detect, key_point, 'Color', [255, 0, 0], 'MarkerType', '*');
    end
    pt1 = boundingboxes(max_id, 1:2);
    pt2 = boundingboxes(max_id, 3:4);
    img_detect = cv.rectangle(img_detect, pt1, pt2, 'Color', [255, 0, 0], 'Thickness', 2);
    score = boundingboxes(max_id, 5);
    score = sprintf('%4f', score);
    img_detect = cv.putText(img_detect, score, pt1, 'Color', [255, 0, 0], 'Thickness', 2);
    cv.imwrite(fullfile(detect_root, image_name), img_detect);
    
    % for crop image
    img_crop = img;
    pt1 = boundingboxes(max_id, 1:2);
    pt2 = boundingboxes(max_id, 3:4);
    center_pt = (pt1 + pt2) / 2.0;
    bbox_size = [pt2(1) - pt1(1), pt2(2) - pt1(2)];
    scale = [3, 5];
    bbox_size = scale.*bbox_size;
    pt1 = center_pt - bbox_size / 2.0;
    pt2 = center_pt + bbox_size / 2.0;
    % img_crop = cv.rectangle(img_crop, pt1, pt2, 'Color', [255, 0, 0], 'Thickness', 2);
    img_crop = imcrop(img_crop, [pt1(1), pt1(2), bbox_size(1), bbox_size(2)]);
    [crop_height, crop_width, channel] = size(img_crop);
    if crop_height < 200 && crop_width < 100
      continue;
    end 
    cv.imwrite(fullfile(crop_root, image_name), img_crop);


        
    % for align image
    img_align = img;
    facial5points = [points(1:5, max_id)'; points(6:10, max_id)'];
    facial5points = double(facial5points);
    Tfm = cp2tform(facial5points', coord5points', 'similarity');
    img_align = imtransform(img_align, Tfm, 'XData', [1, crop_size(2)], ...
      'YData', [1, crop_size(1)], 'Size', crop_size);
    cv.imwrite(fullfile(align_root, image_name), img_align);
end
