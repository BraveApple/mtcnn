clear;
mode = 'GPU';
show_detect = true;

% add search path for caffe
addpath('/home/wang/work/caffe/matlab');
% add search path for mexopencv
addpath('/home/wang/disk_4T/wang_data/software/mexopencv');
% add search path for function
addpath('./function');

data_root = '/home/wang/disk_4T/wang_data/original_dataset/CelebA';
output_root = fullfile(data_root, 'Output')
anno_root = fullfile(data_root, 'Anno');
input_root = fullfile(data_root, 'Img/img_celeba');
detect_root = fullfile(output_root, 'detect_image');
align_root = fullfile(output_root, 'align_image');

if ~exist(data_root, 'dir')
  fprintf('Not found data_dir --> %s\n', data_root);
  return;
end

if ~exist(anno_root, 'dir')
  fprintf('Not found anno_dir --> %s\n', anno_root);
  return;
end

if ~exist(input_root, 'dir')
  fprintf('Not found input_root --> %s\n', input_root);
  return;
end

if exist(output_root, 'dir')
  fprintf('Found output_root --> %s, so we remove it\n', output_root);
  rmdir(output_root, 's');
end
mkdir(output_root);

if exist(detect_root, 'dir')
  fprintf('Found detect_root --> %s, so we remove it\n', detect_root);
  rmdir(detect_root, 's');
end
mkdir(detect_root);

if exist(align_root, 'dir')
  fprintf('Found align_root --> %s, so we remove it\n', align_root);
  rmdir(align_root, 's');
end
mkdir(align_root);

input_label_txt = fullfile(anno_root, 'list_attr_celeba_tmp.txt');
output_label_txt = fullfile(output_root, 'output_label.txt');
no_face_txt = fullfile(output_root, 'no_face.txt');

if ~exist(input_label_txt, 'file')
  fprintf('Not found file --> %s\n', input_label_txt);
  return;
end

output_label_fid = fopen(output_label_txt, 'w');
no_face_fid = fopen(no_face_txt, 'w');

% crop_size = [112, 96];
% coord5points = [30.2946, 65.5318, 48.0252, 33.5493, 62.7299; ...
%                 51.6963, 51.5014, 71.7366, 92.3655, 92.2041];

crop_size = [256, 256];
coord5points = [110, 158, 134, 112, 152; ...
                100, 100, 124, 152, 152];
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

input_label_data = importdata(input_label_txt);
img_list = input_label_data.rowheaders;
labels = input_label_data.data;

for i = 1:length(img_list)

  image_name = img_list{i};
  % sprintf('image_name = %s', image_name)
  image_path = fullfile(input_root, image_name);
  fprintf('image_path = %s\n', image_path);
    
  if ~exist(image_path, 'file')
      fprintf('Not found file --> %s\n', image_path);
      return;
  end
  
  img = [];
  try
    img = cv.imread(image_path);
  catch
    warning('Fail to read image_path --> %s\n', image_path);
    fprintf(no_face_id, '%s --> Fail to read image\n', image_name);
    continue;
  end
	  
  % we recommend you to set minsize as x * short side
	% minl=min([size(img,1) size(img,2)]);
	% minsize=fix(minl*0.1)
    
  tic
  [boundingboxes, points] = detect_face(img, minsize, PNet, RNet, ONet, LNet, threshold, false, factor);
	toc
	
	numbox = size(boundingboxes, 1);
    
  if numbox == 0
    fprintf('%s --> numbox == 0\n', image_name);
    fprintf(no_face_fid, '%s --> numbox == 0\n', image_name);
    continue;
  end
  
  [max_score, max_id] = max(boundingboxes(:, 5)); 
  
  % for detect image
  if show_detect
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
    detect_img_path = fullfile(detect_root, image_name);
    [root, name, ext] = fileparts(detect_img_path);
    if ~exist(root, 'dir')
      mkdir(root);
    end
    cv.imwrite(detect_img_path, img_detect); 
 end
  
  if max_score < 0.9
    fprintf('%s --> max_score < 0.9\n', image_name);
    fprintf(no_face_fid, '%s --> max_score < 0.9\n', image_name);
    continue;
  end
         
  % for align image
  img_align = img;
  facial5points = [points(1:5, max_id)'; points(6:10, max_id)'];
  facial5points = double(facial5points);
  Tfm = cp2tform(facial5points', coord5points', 'similarity');
  img_align = imtransform(img_align, Tfm, 'XData', [1, crop_size(2)], ...
    'YData', [1, crop_size(1)], 'Size', crop_size);
  align_img_path = fullfile(align_root, image_name);
  [root, name, ext] = fileparts(align_img_path);
  if ~exist(root, 'dir')
    mkdir(root);
  end
  cv.imwrite(align_img_path, img_align);
  fprintf(output_label_fid, '%s %s\n', image_name, num2str(labels(i, :)));
  
  fprintf('\n');
end

fclose(output_label_fid);
fclose(no_face_fid);
