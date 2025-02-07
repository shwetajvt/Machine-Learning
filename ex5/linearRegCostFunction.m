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
%===============cost calculation===================================== 
   hypothesis = X * theta;
   J_unreg = (1/(2*m)) * (sum((hypothesis - y) .^2));
   regularization_term = (lambda/(2*m)) * (sum(theta(2:end,:) .^2));
   J = J_unreg + regularization_term;
%================gradient calculation=================================
   grad_unreg = (1/m) * (X' * (hypothesis - y));   %2x1
   reg_term = (lambda ./ m) * (theta(2:end,:));       %1x1
   grad_reg =  grad_unreg(2:end,:) + reg_term;  
   grad = [grad_unreg(1); grad_reg]; 

% =========================================================================

grad = grad(:);

end
