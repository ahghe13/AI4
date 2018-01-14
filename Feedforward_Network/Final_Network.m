% AI4 project
% File description: Constructs the Final Network
% Student: Ahmad Gheith
% Supervisor: John Hallam

clear;
rand('seed', 42);

% Read data
data_file = importdata('../Data/BTC_data.csv');
data = data_file.data;
[N, M] = size(data);

% Set window size and select sequence C
w_size = 10;
C = [1 3];

% Arrange the data
[input, target] = ArrangeData(data, C, w_size);

% Construct and train a network
[net, mse] =  NetPerf(input, target);

% Print MSE
disp(['MSE = ', num2str(mse)]);


Yh = net(input)'; Yh = padarray(Yh, w_size, 'pre');
Y = target'; Y = padarray(Y, w_size, 'pre');

% Plotting the performance of the Final Network
figure;

subplot(1,2,1);
plot(Y);
hold on;
plot(Yh);
xlim([1856 N])
title('Performance of FFNN: Full testing period');
xlabel('Day');
ylabel('Price [USD]');
legend('Actual price series', 'Predicted price series');

subplot(1,2,2)
plot(Y);
hold on;
plot(Yh);
xlim([1900 1950])
title('Performance of FFNN: Subperiod of testing period');
xlabel('Day');
ylabel('Price [USD]');
legend('Actual price series', 'Predicted price series');
