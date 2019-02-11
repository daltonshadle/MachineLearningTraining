% This is my code for exercise 3

%% ******************** Load saved matrices from file *********************
load('ex3data1.mat');
% The matrices X and y will now be in your MATLAB environment


%% **************************** Visualize Data ****************************
m = size(X, 1);
% Randomly select 100 data points to display
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);

displayData(sel);


%% ******************** Regularized Logitic Regression ********************
theta_t = [-2; -1; 1; 2];
X_t = [ones(5,1) reshape(1:15,5,3)/10];
y_t = ([1;0;1;0;1] >= 0.5);
lambda_t = 3;
[J, grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t);

fprintf('Cost: %f | Expected cost: 2.534819\n', J);
fprintf('Gradients: '); 
fprintf('%f, ', grad);
fprintf('\nExp. grad.: 0.146561, -0.548558, 0.724722, 1.398003\n');


%% ****************************** One vs All ******************************
num_labels = 10; % 10 labels (or classifications), from 1 to 10 
lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);


%% ************************** Predict One vs All **************************
pred = predictOneVsAll(all_theta, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
fprintf('\nExp. Training Accuracy: 94.9\n');


%% ************************* Neural Network Stuff *************************
load('ex3data1.mat');
m = size(X, 1);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);
displayData(X(sel, :));

% Load saved matrices from file
load('ex3weights.mat'); 
% Theta1 has size 25 x 401
% Theta2 has size 10 x 26


%% *************************** Predict (for NN) ***************************
pred = predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
fprintf('\nExp. Training Accuracy: 97.5\n');

%% *************************** Predict More Stuff ***************************
%  Randomly permute examples
rp = randi(m);
% Predict
pred = predict(Theta1, Theta2, X(rp,:));
fprintf('\nNeural Network Prediction: %d (digit %d)\n', pred, mod(pred, 10));
% Display 
displayData(X(rp, :));  
