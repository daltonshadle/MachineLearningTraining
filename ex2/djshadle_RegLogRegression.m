% this is my code for exercise 2 - regularized logistic regression

%% ************************* Initializing Data ************************* %%
% Load Data
%  The first two columns contains the X values and the third column
%  contains the label (y).
data = load('ex2data2.txt');
X = data(:, [1, 2]); y = data(:, 3);

%  Setup the data matrix appropriately
[m, n] = size(X);

%% *************************** Plotting Data *************************** %%
%Generating a new figure
f1 = figure();
hold on

%Plotting data using plotData function, ignoring first row of 1's in X
plotData(X, y);

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
% Specified in plot order
legend('y = 1', 'y = 0')
hold off;


%% ********************** Add Polynomial Features ********************** %%
% !!! NOTE !!! that mapFeature also adds a column of ones for us, so 
% the intercept term is handled
X = mapFeature(X(:,1), X(:,2));


%% ********************* Regularized Log Regression ******************** %%
% Compute and display initial cost and gradient for regularized logistic regression

% Re-initialize the fitting parameters, since X was feature mapped
initial_theta = zeros(size(X, 2), 1);
lambda = 1;

[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);
fprintf('Cost at initial theta (zeros): %f\n', cost);

%% **************************** Optimizing ***************************** %%
% Able to vary lambda here to see its affect on fitting
lambda = 0.5;

% Set Options
options = optimoptions(@fminunc,'Algorithm','Quasi-Newton','GradObj', 'on', 'MaxIter', 1000);

% Optimize
[theta, J, exit_flag] = fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);


%% ****************************** Plotting ***************************** %%
% Plot Boundary
f2 = figure();
plotDecisionBoundary(theta, X, y);
hold on;
title(sprintf('lambda = %g', lambda))

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

legend('y = 1', 'y = 0', 'Decision boundary')
hold off;

% Compute accuracy on our training set
p = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);



