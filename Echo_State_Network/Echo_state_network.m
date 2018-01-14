% AI4 project
% File description: Echo State Network
% Student: Ahmad Gheith
% Supervisor: John Hallam
% Inspired by Mantas Lukosevicius' implementation: http://minds.jacobs-university.de/mantas/code

clear;
rand('seed', 42);

% Load data
data_file = importdata('../Data/BTC_data.csv');
data = data_file.data;
[N, M] = size(data);

% Split the available data into 3 sets: Initialization, training, and testing sets
d_init = 50;                        % Initialization set
d_train = floor((N-d_init)*0.4);     % Training set: Used for training the weights
d_test = floor((N-d_init)*0.6);      % Testing set: Used to test the network

% Declare the ESN parameters
n_in = 2;
n_res = 300;
n_out = 1;
a = 0.5;            % leaking rate

% Initialize Win and W
Win = (rand(n_res,1+n_in)-0.5);
W = rand(n_res,n_res)-0.5;

% Normalizing by using spectral radius
opt.disp = 0;
rhoW = abs(eigs(W,1,'LM',opt));
W = W .* (1.25/rhoW);

% Allocate memory for the internal state matrix, X
X = zeros(1+n_in+n_res,d_train);

% Run the reservoir with the data and compute X
x = zeros(n_res,1)+1;
for t = 1:d_init+d_train
    u = data(t,:)';
    x = (1-a)*x + a*tanh(Win*[1;u] + W*x);
    if t > d_init
        X(:,t-d_init) = [1;u;x];
    end
end

% Set the corresponding target matrix directly
Y = data(d_init+2:d_init+d_train+1);

% Train the Network
Wout = Y*pinv(X);

% Compute predictions
Yh = zeros(n_out,d_test);
u = data(d_init+d_train+1,:)';
for t = 1:d_test 
    x = (1-a)*x + a*tanh(Win*[1;u] + W*x);
    yh = Wout*[1;u;x];
    Yh(:,t) = yh;
    u = data(d_init+d_train+1+t,:)';
end

% Compute MSE
Y = data(d_init+d_train+2:d_init+d_train+d_test+1);
mse = (sum((Y-Yh).^2))./d_test;

disp(['MSE = ', num2str(mse)]);

% Save the network
net.Yh = Yh;
net.Win = Win;
net.W = W;
net.Wout = Wout;
net.mse = mse;

Y = data(:,1);
Yh = Yh'; Yh = padarray(Yh, N-d_test, 'pre');

% Plotting the performance of the Final Network
figure;

subplot(1,2,1);
plot(Y);
hold on;
plot(Yh);
xlim([N-d_test+1 N])
title('Performance of ESN: Full testing period');
xlabel('Day');
ylabel('Price [USD]');
legend('Actual price series', 'Predicted price series');

subplot(1,2,2)
plot(Y);
hold on;
plot(Yh);
xlim([1900 1950])
title('Performance of ESN: Subperiod of testing period');
xlabel('Day');
ylabel('Price [USD]');
legend('Actual price series', 'Predicted price series');
