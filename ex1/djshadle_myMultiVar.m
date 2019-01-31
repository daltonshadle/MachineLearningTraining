%this is my code for exercise 1

%************************** Initializing Data ****************************%
data = load('ex1data2.txt'); % read comma separated data
X = data(:, 1:2); % matrix X for the input train data from col 1,2 of data
y = data(:, 3); % vector y for the output training data from col 3 of data
m = length(y); % number of training examples

theta = zeros(3, 1); % vector theta intialized to 0
num_iters = 400; % number of iterations in the learning algorithm
alpha = 0.1; % alpha is the learning step size

% Scale features and set them to zero mean
[X, mu, sigma] = featureNormalize(X);

% performing matrix operations on X
X = [ones(m, 1) X]; % Add intercept term to X

%*************************** Computing Cost ******************************%

fprintf('Computed cost for theta initialized to 0: %f\n', computeCostMulti(X,y,theta));

%********************** Running Gradient Descent *************************%
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
fprintf('Computed cost after gradient descent: %f\n', computeCostMulti(X,y,theta));

% Plot of cost function while running through gradient descent
t = 1:1:num_iters;
plot (t, J_history);

% Display gradient descent's result
fprintf('Theta 1 computed from gradient descent: %f\n',theta(1));
fprintf('Theta 2 computed from gradient descent: %f\n',theta(2));
fprintf('Theta 3 computed from gradient descent: %f\n',theta(3));

% Estimate the price of a 1650 sq-ft, 3 br house
house_size = 1650;
house_br = 3;

house_size_N = (house_size - mu(1,1)) / sigma(1,1);
house_br_N = (house_br - mu(1,2)) / sigma(1,2);

price = theta(1) + (theta(2) * house_size_N) + (theta(3) * house_br_N);
fprintf('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): $%.2f\n', price);





