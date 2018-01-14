% AI4 project
% File description: Echo State Network
% Student: Ahmad Gheith
% Supervisor: John Hallam
% Inspired by Mantas Lukosevicius' implementation: http://minds.jacobs-university.de/mantas/code

clear;

% Load data
data_file = importdata('../Data/BTC_data.csv');
data = data_file.data;
[N, M] = size(data);

% Split the available data into 3 sets: Initialization, training, and testing sets
d_init = 150;                        % Initialization set
d_train = floor((N-d_init)*0.5);     % Training set: Used for training the weights
d_test = floor((N-d_init)*0.5);      % Testing set: Used to test the network

% Declare the ESN parameters
n_in = 2;
n_res = 500;
n_out = 1;
a = 0.4;            % leaking rate

% Declare a struct for storing the Network
net.mse = 10000000;

rand('seed', 35);
rep = 100;

for n=1:rep
    Win = (rand(n_res,1+n_in)-0.5);
    W = rand(n_res,n_res)-0.5;

    % Normalizing by using spectral radius
    opt.disp = 0;
    rhoW = abs(eigs(W,1,'LM',opt));
    W = W .* (1.25/rhoW);

    % Allocating memory for the internal state matrix, X
    X = zeros(1+n_in+n_res,d_train-d_init);

    % Run the reservoir with the data and compute X
    x = zeros(n_res,1)+1;
    for t = 1:d_train
        u = data(t,:)';
        x = (1-a)*x + a*tanh(Win*[1;u] + W*x);
        if t > d_init
            X(:,t-d_init) = [1;u;x];
        end
    end

    % Set the corresponding target matrix directly
    Y = data(d_init+2:d_train+1);

    % Train the Network
    Wout = Y*pinv(X);

    % Compute predictions
    Y_hat = zeros(n_out,d_test);
    u = data(d_train+1,:)';
    for t = 1:d_test 
        x = (1-a)*x + a*tanh(Win*[1;u] + W*x);
        y = Wout*[1;u;x];
        Y_hat(:,t) = y;
        u = data(d_train+t+1,:)';
    end
    
    % Compute MSE
    mse = sum((data(d_train+2:d_train+d_test+1)-Y_hat(1,1:d_test)).^2)./d_test;
    
    % Check if new MSE is better than the previous
    if mse < net.mse
        disp(['Better network found with MSE ', num2str(mse)]);
        net.mse = mse;
        net.Win = Win;
        net.W = W;
        net.Wout = Wout;
        net.Y_hat = Y_hat;
    end
    
    disp(['Iteration ', num2str(n), ' completed']);
end


% plot some signals
figure;
plot(data(d_train+2:d_train+d_test+1));
hold on;
plot( net.Y_hat');
title('Predicted vs. Actual Bitcoin Price Series');
xlabel('Day');
ylabel('Price [USD]');
legend('Actual Price', 'Predicted Price');
hold off;
