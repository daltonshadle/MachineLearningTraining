%this is my code for exercise 1

%************************** Initializing Data ****************************%
data = load('ex1data1.txt'); % read comma separated data

X = data(:, 1); % matrix X for the input training data from col 1 of data
y = data(:, 2); % vector y for the output training data from col 2 of data
theta = [0; 0]; % vector theta intialized to 0
iterations = 1500; % number of iterations in the learning algorithm
alpha = 0.01; % alpha is the learning step size

%Initializing figure and plotting info
f1 = figure;

% plot X,y on scatter plot
plotData(X,y);

% performing matrix operations on X
m = length(X); % number of training examples
X = [ones(m, 1), data(:,1)]; % Add a column of ones to x


%*************************** Computing Cost ******************************%
% Compute and display initial cost with theta all zeros
fprintf('Computed cost of theta initialized to 0: %f\n', computeCost(X, y, theta));


%********************** Running Gradient Descent *************************%
% Compute theta
theta = gradientDescent(X, y, theta, alpha, iterations);

% Print theta to screen
% Display gradient descent's result
fprintf('Theta 1 computed from gradient descent: %f\n',theta(1));
fprintf('Theta 2 computed from gradient descent: %f\n',theta(2));

% Plot the linear fit
hold on; % keep previous plot visible
plot(X(:,2), X*theta, '-')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure

% Predict values for population sizes of 35,000 and 70,000
predict1 = [1, 3.5] * theta;
fprintf('For population = 35,000, we predict a profit of $%.2f\n', predict1*10000);

predict2 = [1, 7] * theta;
fprintf('For population = 70,000, we predict a profit of $%.2f\n', predict2*10000);

%************************ Visualizing J(theta) ***************************%
% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];    
	  J_vals(i,j) = computeCost(X, y, t);
    end
end

% Because of the way meshgrids work in the surf command, we need to 
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';


% Surface plot
f2 = figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

% Contour plot
f3 = figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
hold off;
