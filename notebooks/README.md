<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Week 1:
- [MATLAB README](https://github.com/T-Mosher/ml-support/blob/main/MATLAB_README.pdf)

# Week 2
Cost function: $$ J(\theta) = \frac{1}{2m} \sum \limits_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2 $$
Hypothesis for linear regression: $$ h\theta(x) = \theta^Tx = \theta_0 + \theta_1x1 $$

### costFunction.m
```
function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

grad = zeros(size(theta));

% You need to return the following variables correctly 
%term1 = y*log(sigmoid(X));
%term2 = (1 - y)*log(1- sigmoid(X));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

term1 =  -1 * (y .* log(sigmoid(X * theta)));
term2 = (1 - y) .* log(1 - sigmoid(X * theta));

J = sum(term1 - term2) / m;

grad = (X' * (sigmoid(X * theta) - y)) * (1/m);

% grad = 1/m * sum((sigmoid(X) - y)*X);

% =============================================================

end
```

### costFunctionReg.m
```
function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%term1 =  -1 * (y .* log(sigmoid(X * theta)));
%term2 = (1 - y) .* log(1 - sigmoid(X * theta));
%term3 = (lambda/(2*m))*sum(theta.^2);
%J = (sum(term1 - term2) / m) + term3;
%grad = (X' * (sigmoid(X * theta) - y)) * (1/m) + (lambda/m)*theta;

J=(1/m) * sum((-y .* log(sigmoid(X*theta)))- ((1-y) .* log(1-sigmoid(X*theta))));
J=J+ (lambda/(2*m)) * sum(theta(2:size(theta)).*theta(2:size(theta)));


grad(1)=(sum(((sigmoid(X*theta)-y).* X(:,1)))/ m);

for j=2:size(theta)
    grad(j)=(sum(((sigmoid(X*theta)-y).* X(:,j)))/ m)+(lambda * theta(j)/m);
end

% =============================================================

end
```

### gradientDescent.m
```
function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
%    theta = theta - (alpha/m)*( sum((X*theta - y).*X));
%    fprintf('theta %d', length(theta))
% https://yuting3656.github.io/yutingblog/octave2python/ex1-gradient-descent
    x = X(:,2);
    h = theta(1) + (theta(2)*x);

    theta_zero = theta(1) - alpha * (1/m) * sum(h - y);
    theta_one  = theta(2) - alpha * (1/m) * sum((h - y) .* x);

    theta = [theta_zero; theta_one];

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
```

# Week 3
- Regularization
