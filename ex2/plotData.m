function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

 admitted_student = find(y==1);
 notadmitted_student = find(y==0);

 plot(X(admitted_student,1),X(admitted_student,2), 'k+', 'LineWidth', 4, 'MarkerSize', 10);

 plot(X(notadmitted_student,1), X(notadmitted_student,2), 'ko','MarkerFaceColor', 'm', 'MarkerSize', 10);









% =========================================================================



hold off;

end
