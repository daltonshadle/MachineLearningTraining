function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
    % Initializing hypothesis
    hypo = X * theta;
    % Removing the first row in theta since we don't regularize theta_0
    reg_theta = theta;
    reg_theta([1],:) = [];
    
    % ************************** Calculating Cost *************************
    % Calculating J_part1 which is the cost without regularization
    J_part1 = (1 / (2 * m)) * sum((hypo - y) .^ 2);
    
    % Calculating J_part2 which is the regularization cost;
    J_part2 = (lambda / (2 * m)) * sum (reg_theta .^ 2);
    
    % Calculating overall cost J
    J = J_part1 + J_part2;
    
    % ************************ Calculating Gradient *********************** 
    % Calculating grad_part1 which is the gradient without regularization
    grad_part1 = (1 / m) * X' * (hypo - y);
    
    % Calculating grad_part2 which is the regularization gradient
    grad_part2 = (lambda / m) .* reg_theta;
    % Add extra column for theta_0
    grad_part2 = [0; grad_part2];
    
    % Calculating overall gradient grad
    grad = grad_part1 + grad_part2;
    
% =========================================================================

grad = grad(:);

end
