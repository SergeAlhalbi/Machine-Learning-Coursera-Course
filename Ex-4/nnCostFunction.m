function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Written by me
a1 = [ones(m,1) ,X];


% Start of Part 1: ---------------------------------------------------------

z2 = Theta1 * (a1)';
pre1_a2 = sigmoid (z2);
pre2_a2 = [ones(m, 1), (pre1_a2)'];
a2 = (pre2_a2)'; 

pre1_a3 = Theta2 * a2;
pre2_a3 = sigmoid (pre1_a3);

h = a3 = (pre2_a3)';

% I found h (in the course case: it's 5000 x 10)


A = 0;

for i = 1:m


	B = zeros(num_labels, 1);

	for k = 1:num_labels
		
		B(k) = y(i) == k;

		% B is the new vectorized y in the correct style needed to compute the cost function

	end


u = -B' * log( (h(i, :))' )   -   (1 - B') * log( (1 - (h(i, :))') );

% u is a 1 x 1 scalar: sum of h_k's of 1 example (ith example)

A = A + u;

% A is a 1 x 1 scalar: sum of h_k's of all m examples

J0 = ((1 / m) * A);

% J0 is the unregularized cost function

J = J0 + (lambda / (2*m)) * (  sum(sum((Theta1(:, 2:end)).^2))  +  sum(sum((Theta2(:, 2:end)).^2))  );


end

% Part 1: end --------------------------------------------------------------

DELTA1 = zeros(size(Theta1));
DELTA2 = zeros(size(Theta2));

% Doing it for one example as suggested in the course (There's an advanced fully vectorized way to do this as well)

for t = 1:m
	

	th_a1 = (a1(t, :))';
	
	th_z2 = Theta1 * th_a1;
	pre1_th_a2 = sigmoid(th_z2);
	th_a2 = [1; pre1_th_a2];

	th_z3 = Theta2 * th_a2;
	th_a3 = sigmoid(th_z3);


	B = zeros(num_labels, 1);

	for k = 1:num_labels
		
			B(k) = y(t) == k;

			% B is the new vectorized y in the correct style needed to compute the cost function

	end

	delta3 = th_a3 - B;

	delta2 = (Theta2(:, 2:end))' * delta3 .* sigmoidGradient(th_z2);

	
	DELTA1 = DELTA1 + delta2 * (th_a1)';

	DELTA2 = DELTA2 + delta3 * (th_a2)';


end


Theta1_grad = (1/m) * ( (DELTA1) + (lambda) .* [zeros(size(Theta1, 1), 1), Theta1(:, 2:end)] );
Theta2_grad = (1/m) * ( (DELTA2) + (lambda) .* [zeros(size(Theta2, 1), 1), Theta2(:, 2:end)] );


% Unrolling
grad = [Theta1_grad(:); Theta2_grad(:)];

% Written by me: DO NOT CHANGE THE " Theta1_grad " and " Theta2_grad " NAMES, FOR SOME REASON IT DOES'S NOT WORK











% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end



