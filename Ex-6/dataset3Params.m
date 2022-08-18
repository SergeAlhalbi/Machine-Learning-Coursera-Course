function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_variations = 8;
sigma_variations = 8;
iter_nb = sigma_variations*C_variations;
interval = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';
Accuracy_percentage_for_all_models = zeros(iter_nb, 1);

p = 1;

for i = 1:C_variations

	for j = 1:sigma_variations

		C = interval(i);

		sigma = interval(j);

		% Now, learn the model (parameters theta that are hidden in these cases)	

		model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));

		% Then, evaluate on the cross-validation set

		predictions = svmPredict(model, Xval);
	
		Accuracy_percentage_for_the_model = mean(double(predictions == yval)) * 100;

		Accuracy_percentage_for_all_models(p++) = Accuracy_percentage_for_the_model;

	end

end


[Most_accurate_model_percentage, Index] = max(Accuracy_percentage_for_all_models);


% Build the matrix to help order the iterations

Matrix = zeros(iter_nb,2);
k = 1;

for i = 1:C_variations

	for j = 1:sigma_variations

		C = interval(i);

		sigma = interval(j);

		Matrix(k++, :) = [C, sigma];
			
	end
end

C = Matrix(Index,1);
sigma = Matrix(Index,2);


% =========================================================================

end
