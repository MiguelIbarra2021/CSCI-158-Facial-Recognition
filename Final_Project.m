%% Step 0: Initalize Workspace
close all % close figures and other stuff
clear % clean the worksapce
clc % clear command line

%% Step 1: Load Dataset
file_pattern = fullfile('img','*.gif');
img_files = dir(file_pattern);

num_images = numel(img_files);
imgs = cell(1, num_images);

% Load into an array of images
for i = 1: num_images
    img_name = img_files(i).name;
    fullPath = fullfile('img', img_name);
    raw_img = imread(fullPath);
    imgs{i} = im2double(raw_img);
end

%% Step 2: Process Images

% Image size
n1 = 100;
n2 = 100;

for i = 1: num_images
    imgs{i} = PreprocessImage(imgs{i});
end

%% Step 3: Feature Extraction

% Allocate space for data matrix
img_matrix = zeros(num_images, 10000);

for i = 1: num_images
    img_matrix(i, :) = imgs{i}(:);
end

% Center the data
mean_face = mean(img_matrix, 2);
faces_centered = img_matrix - mean_face;

% Compute covariance matrix
covariance_matrix = faces_centered' * faces_centered / (num_images - 1);

% Calculate eigenvectors and eigenvalues
[eigen_vectors, eigen_values] = eig(covariance_matrix, 'vector');

% Sort eigenvalues
[eigen_values, sort_index] = sort(eigen_values, 'descend'); % Done in descending order
eigen_vectors = eigen_vectors(:, sort_index);

% Normalize
eigen_faces = faces_centered * eigen_vectors;
eigen_faces = eigen_faces ./ vecnorm(eigen_faces);

% Display Mean Face
figure(4);
imagesc(reshape(mean_face, n1, n2));
colormap gray; axis off; title('Mean Face');

%% Functions

% Preprocess Image Input
function img = PreprocessImage(input)
    % Convert image to greyscale
    grey_img = im2gray(input);

    % Using premade facial detection algorithm
    detector = vision.CascadeObjectDetector();
    bound_box = step(detector, grey_img); % Contains X, Y, width, & height
    
    % Simply display the bounding box over the image
    figure(1);
    boxed_image = insertShape(grey_img, 'Rectangle', bound_box);
    imshow(boxed_image);

    % Crop image
    figure(2);
    cropped_img = imcrop(grey_img, bound_box);
    imshow(cropped_img);

    % Resize image
    figure(3);
    img = imresize(cropped_img,[100, 100]);
    imshow(img);
end