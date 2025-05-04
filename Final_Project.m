%% Step 0: Initalize Workspace
close all % close figures and other stuff
clear % clean the worksapce
clc % clear command line

%% Step 1: Load Dataset for training, validation, and testing
train_pattern = fullfile('Images/train/','*.png');
train_img_files = dir(train_pattern);

train_num_images = numel(train_img_files);
train_imgs = cell(1, train_num_images);

% Load into an array of images
for i = 1: train_num_images
    img_name = train_img_files(i).name;
    fullPath = fullfile('Images/train', img_name);
    raw_img = imread(fullPath);
    train_imgs{i} = im2double(raw_img);
end

%% Step 2: Process Images

% Image size
n1 = 100;
n2 = 100;

for i = 1: train_num_images
    train_imgs{i} = PreprocessImage(train_imgs{i});
end

%% Step 3: Feature Extraction

% Allocate space for data matrix
img_matrix = zeros(train_num_images, 10000);

for i = 1: train_num_images
    img_matrix(i, :) = train_imgs{i}(:);
end

% Center the data
mean_face = mean(img_matrix, 1);
faces_centered = img_matrix - mean_face;

% Compute covariance matrix
covariance_matrix = faces_centered' * faces_centered / (train_num_images - 1);

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

% Choose components to be used
K = 50;
evecs_reduced = eigen_vectors(:, 1:K);           % 10000×50
projected_data = faces_centered * evecs_reduced; % num_images×50

labels = zeros(train_num_images,1);
for i = 1:train_num_images
    name = train_img_files(i).name;
    tokens = regexp(name, '([a-z_]+)(\d+)\.png$', 'tokens');
    labels(i) = str2double(tokens{1}{2});
end

%% Step 4: Split Data into Training and Test Sets

% Randomize and split (80% train, 20% test)
rng(1); % for reproducibility
indices = randperm(train_num_images);
train_idx = indices(1:round(0.8*train_num_images));
test_idx  = indices(round(0.8*train_num_images)+1:end);

X_train = projected_data(train_idx, :);
y_train = labels(train_idx);
X_test  = projected_data(test_idx, :);
y_test  = labels(test_idx);

%% Step 5: Train Classifiers

% KNN Model
knn_model = fitcknn(X_train, y_train, 'NumNeighbors', 3);

% Naive Bayes Model
nb_model = fitcnb(X_train, y_train);

%% Step 7: Predict a Custom Face

% Load and preprocess custom image
custom_path = 'images.jpg'; % Replace with your actual file
custom_raw = im2double(imread(custom_path));
custom_processed = PreprocessImage(custom_raw);

% Flatten and project into PCA space
custom_vector = custom_processed(:)' - mean_face;
custom_projected = custom_vector * evecs_reduced;

% Predict with both models
custom_pred_knn = predict(knn_model, custom_projected);
custom_pred_nb  = predict(nb_model, custom_projected);

% Display results
fprintf('KNN Prediction: Subject %02d\n', custom_pred_knn);
fprintf('Naive Bayes Prediction: Subject %02d\n', custom_pred_nb);

figure;
imshow(custom_processed);
title(sprintf('Predicted by KNN: Subject %02d | NB: Subject %02d', ...
              custom_pred_knn, custom_pred_nb));

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

