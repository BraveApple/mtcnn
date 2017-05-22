clear;
mode = 'GPU';

% add search path for caffe
addpath('/home/wang/work/caffe/matlab');
% add search path for mexopencv
addpath('/home/wang/disk_4T/wang_data/software/mexopencv');
% add search path for function
addpath('./function');

data_root = '/home/wang/disk_4T/wang_data/original_dataset/SUN';
anno_root = fullfile(data_root, 'Anno');
img_root = fullfile(data_root, 'Img');

assert(logical(exist(data_root, 'dir')), 'Not found directory data_root --> %s\n', data_root);
assert(logical(exist(anno_root, 'dir')), 'Not found directory anno_root --> %s\n', anno_root);
assert(logical(exist(img_root, 'dir')), 'Not found directory img_root --> %s\n', img_root);

output_root = fullfile(data_root, 'Output');
face_root = fullfile(output_root, 'face_image');
no_face_root = fullfile(output_root, 'no_face_image');

if exist(output_root, 'dir')
  fprintf('Found output_root --> %s, so we remove it\n', output_root);
  rmdir(output_root, 's');
end
mkdir(output_root);

if exist(face_root, 'dir')
  fprintf('Found detect_root --> %s, so we remove it\n', face_root);
  rmdir(face_root, 's');
end
mkdir(face_root);

if exist(no_face_root, 'dir')
  fprintf('Found no_face_root --> %s, so we remove it\n', no_face_root);
  rmdir(no_face_root, 's');
end
mkdir(no_face_root);

list_txt = fullfile(anno_root, 'list_img.txt');
face_txt = fullfile(output_root, 'face.txt');
no_face_txt = fullfile(output_root, 'no_face.txt');

assert(logical(exist(list_txt, 'file')), 'Not found file list_txt --> %s\n', list_txt);

face_fid = fopen(face_txt, 'w');
no_face_fid = fopen(no_face_txt, 'w');

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

img_list = importdata(list_txt);

for i = 1:length(img_list)
  
  fprintf('\n');
  img_subpath = img_list{i};
  img_path = fullfile(img_root, img_subpath);
  fprintf('img_path = %s\n', img_path);
  assert(logical(exist(img_path, 'file')), 'Not found file img_path --> %s\n', img_path);
  
  img = [];
  try
    img = cv.imread(img_path);
  catch
    warning('Fail to read image_path --> %s\n', img_path);
    continue;
  end
	  
  % we recommend you to set minsize as x * short side
	% minl=min([size(img,1) size(img,2)]);
	% minsize=fix(minl*0.1)
    
  tic
  [boundingboxes, points] = detect_face(img, minsize, PNet, RNet, ONet, LNet, threshold, false, factor);
	toc
	
	numbox = size(boundingboxes, 1);
  
  face_img_path = fullfile(face_root, img_subpath);
  [path, name, ext] = fileparts(face_img_path);
  if ~exist(path, 'dir')
    mkdir(path);    
  end
  
  no_face_img_path = fullfile(no_face_root, img_subpath);
  [path, name, ext] = fileparts(no_face_img_path);
  if ~exist(path, 'dir')
    mkdir(path);
  end

  if numbox == 0
    fprintf('%s --> numbox == 0\n', img_subpath);
    fprintf(no_face_fid, '%s\n', img_subpath);
    cv.imwrite(no_face_img_path, img);
    continue;
  end
  
  [max_score, max_id] = max(boundingboxes(:, 5)); 
  
  if max_score < 0.9
    fprintf('%s --> max_score < 0.9\n', img_subpath);
    fprintf(no_face_fid, '%s\n', img_subpath);
    cv.imwrite(no_face_img_path, img);
    continue;
  end

  % for face image
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
  
  fprintf('%s --> find a face\n', img_subpath);
  fprintf(face_fid, '%s\n', img_subpath);
  cv.imwrite(face_img_path, img_detect);
end

fclose(face_fid);
fclose(no_face_fid);
