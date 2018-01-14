% AI4 project
% File description: Constructs, trains, and computes the performance of a Feedforward network
% Student: Ahmad Gheith
% Supervisor: John Hallam

function [network, performance] = NetPerf(input, target)
    n_i = size(input); n_i = n_i(1);
    n_o = size(target); n_o = n_o(1);
    n_h = floor(sqrt(n_i*n_o)+0.5);

    % Construct the network
    net = fitnet(n_h,'trainlm');
    net.trainParam.showWindow = false;

    % Setup Division of Data for Training, Validation, Testing
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 15/100;

    % Train the Network
    [net,tr] = train(net,input,target);

    performance = tr.best_tperf;
    network = net;

end