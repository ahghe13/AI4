% AI4 project
% File description: Repeats SFFS and collects data of the frequency of each outputted sequence
% Student: Ahmad Gheith
% Supervisor: John Hallam

clear;

itr = 500;
w_size = 10;

data_file = importdata('../Data/BTC_data.csv');
data = data_file.data;
[N, M] = size(data);

C_seq = zeros(1,w_size*M)-1;
C_score = [0];

for i=1:itr
    disp(['Iteration ', num2str(i), ' out of ', num2str(itr)])
    C = SFFS(data, w_size);
    C_sort = sort(C);
    [r,c] = size(C_sort);
    C_sort = padarray(C_sort', w_size*M-c, -1, 'pre')';
    C_col = size(C_seq); C_col = C_col(1);
    
    for j=1:C_col
        if isequal(C_sort, C_seq(j,:))
            C_score(j) = C_score(j)+1;
            break;
        elseif j == C_col
            C_seq = [C_seq; C_sort];
            C_score = [C_score; 1];
        end
    end
    
end


% Plotting SFFS statistics
figure;
bar(C_score);
title('SFFS Output Sequences, 500 repetitions');
xlabel('Indices of Sequences, C');
ylabel('Occurring Frequency');