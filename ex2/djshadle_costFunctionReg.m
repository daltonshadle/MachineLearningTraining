function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
len = size(theta);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

    % Computing hypothesis
    hypo = sigmoid(X * theta);
    
    % Computing cost function
    %J_part1 is computing the first portion like first cost function
    J_part1 = (1 / m) * (-y' * log(hypo) - (1 - y') * log(1 - hypo));
    %J_part2 is computing the regularization portion
    J_part2 = (lambda / m) * sum(theta(2:len).^2);
    
    %Add both parts together to return whole cost function
    J = J_part1 + J_part2;
    
    % Computing gradient
    %grad_part1 is computing the first portion like first gradient function
    grad_part1 = (1 / m) * X' * (hypo - y);
    %J_part2 is computing the regularization portion, leaving off theta 0
    grad_part2 = (lambda / m) .* theta(2:len);
    
    %Need to add extra column for theta 0
    grad_part2 = [0; grad_part2];
    
    
    %Add both parts together to return whole gradient
    grad = grad_part1 + grad_part2;


% =============================================================

end
