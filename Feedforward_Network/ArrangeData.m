% AI4 project
% File description: ArrangeData
% Student: Ahmad Gheith
% Supervisor: John Hallam

function [BTC_input, BTC_target] = ArrangeData(data, C, w_size)
    data_r = data';
    data_r = data_r(:);
    
    % Find the size of the data and the window
    [N,M] = size(data);
    
    BTC_input = [];

    % Process the data
    for i=w_size:N-1
        BTC_input = [BTC_input; data_r((i*M)-C)'];
    end
    
    BTC_input = BTC_input';
    BTC_target = data(w_size+1:N,1)';
end

