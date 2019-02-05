% this is my code for exercise 2 - regulated logistic regression

%% ************************* Initializing Data ************************* %%
% Load Data
%  The first two columns contains the X values and the third column
%  contains the label (y).
data = load('ex2data2.txt');
X = data(:, [1, 2]); y = data(:, 3);

%  Setup the data matrix appropriately
[m, n] = size(X);

% Add intercept term to X
X = [ones(m, 1) X];

% Initialize the fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1
lambda = 1;


%% *************************** Plotting Data *************************** %%
%Generating a new figure
f1 = figure();
hold on

%Plotting data using plotData function, ignoring first row of 1's in X
dataX = X(:,[2,3]);
plotData(dataX, y);

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
% Specified in plot order
legend('y = 1', 'y = 0')
hold off;


%% ********************** Add Polynomial Features ********************** %%
% Note that mapFeature also adds a column of ones for us, so the intercept term is handled
X = mapFeature(X(:,1), X(:,2));


%% ********************** Reulated Log Regression ********************** %%
% Compute and display initial cost and gradient for regularized logistic regression

% Re-initialize the fitting parameters since X was feature mapped
initial_theta = zeros(size(X, 2), 1);

[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);
fprintf('Cost at initial theta (zeros): %f\n', cost);