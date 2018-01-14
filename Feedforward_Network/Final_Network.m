% AI4 project
% File description: Constructs the Final Network
% Student: Ahmad Gheith
% Supervisor: John Hallam

clear;

% Read data
data_file = importdata('../Data/BTC_data.csv');
data = data_file.data;

% Set window size and select sequence C
w_size = 10;
C = [1 3];

% Arrange the data
[input, target] = ArrangeData(data, C, w_size);

% Create and train rep Networks and keep the best one as the Final Network
rep = 500;
perf = 1000000;

for i=1:rep
    [net_tmp, perf_tmp] =  NetPerf(input, target);
    if perf_tmp < perf
        perf = perf_tmp;
        net = net_tmp;
        disp(['Better Network found with MSE ', num2str(perf) , '! Iteration ', num2str(i), ' out of ', num2str(rep)])
    end
end

yh = net(input)';
y = target';

% Plotting the performance of the Final Network
figure;
plot(y);
hold on;
plot(yh);
title('Predicted vs. Actual Price Series');
xlabel('Day');
ylabel('Price [USD]');
legend('Target signal', 'Predicted signal');
hold off;

