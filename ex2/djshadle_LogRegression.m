% this is my code for exercise 2 - logistic regression

%% ************************* Initializing Data ************************* %%
% Load Data
% The first two columns contain the exam scores and the third column contains the label
data = load('ex2data1.txt');

%Initializing data to X matrix and y vector
X = data(:, [1, 2]); % matrix X for the input training data from col 1 of data
y = data(:, 3); % vector y for the output training data from col 2 of data

%  Setup the data matrix appropriately
[m, n] = size(X);

% Add intercept term to X
X = [ones(m, 1) X];

% Initialize the fitting parameters
initial_theta = zeros(n + 1, 1);


%% ********************* Plotting Training Data ************************ %%
%Initializing figure and plotting info
f1 = figure;

% Plot the data with + indicating (y = 1) examples and o indicating (y = 0) examples.
hold on;
dataX = X(:,[2,3]);
plotData(dataX, y);
hold off;

% Labels and Legend
xlabel('Exam 1 score');
ylabel('Exam 2 score');

% Specified in plot order
legend('Admitted', 'Not admitted');


%% ********************* Testing Sigmoid Function ********************** %%
fprintf('Sigmoid function at 100: %f\n', sigmoid(100));
fprintf('Sigmoid function at 1: %f\n', sigmoid(1));
fprintf('Sigmoid function at 0: %f\n', sigmoid(0));
fprintf('Sigmoid function at -1: %f\n', sigmoid(-1));
fprintf('Sigmoid function at -100: %f\n', sigmoid(-100));


%% ************************** Cost Function **************************** %%
% Compute and display the initial cost and gradient
[cost, grad] = costFunction(initial_theta, X, y);
fprintf('Cost at initial theta (zeros): %f\n', cost);
disp('Gradient at initial theta (zeros): ');
disp(grad);


%% *********************** Optimizing Function ************************* %%
%  Set options for fminunc
options = optimoptions(@fminunc,'Algorithm','Quasi-Newton','GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

% Print theta
fprintf('Cost at theta found by fminunc: %f\n', cost);
disp('theta:');disp(theta);

% Plot Boundary
f2 = figure();
hold on;
plotDecisionBoundary(theta, X, y);

% Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')
% Specified in plot order
legend('Admitted', 'Not admitted')
hold off;

%% ********************** Evaluating Prediction ************************ %%
%  Predict probability for a student with score 45 on exam 1  and score 85 on exam 2 
prob = sigmoid([1 45 85] * theta);
fprintf('For a student with scores 45 and 85, we predict an admission probability of %f\n\n', prob);

% Compute accuracy on our training set
p = predict(theta, X);
disp(p);
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);

