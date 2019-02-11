function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%



hypothesis = X * Theta';
hypo_alluser = hypothesis .* R;

J = (1/(2)* sum(sum((hypo_alluser - Y) .^ 2)))...
        + ((lambda/2) * sum(sum(Theta.^2)))+((lambda/2)*(sum(sum(X.^2))));


%=======================================
for(i = 1:num_movies)
idx = find((R(i,:)==1));   % all users who rated movie i
theta_rated = Theta(idx,:);
Y_rated = Y(i,idx);
X_rated = X(i,:);
X_grad(i,:) = (((X_rated*theta_rated') -Y_rated) * theta_rated)+(lambda * X_rated);

end

for (j = 1:num_users)
id = find((R(:,j)==1));    % all movies rated by user j
theta_grad_rated = Theta(j,:);
Y_grad_rated = Y(id,j);
X_grad_rated = X(id,:);
Theta_grad(j,:) = (((X_grad_rated * theta_grad_rated') - Y_grad_rated)' * X_grad_rated)...
 +(lambda *theta_grad_rated);

end













% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
