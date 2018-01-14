% AI4 project
% File description: Sequential Forward Feature Selection
% Student: Ahmad Gheith
% Supervisor: John Hallam

function [C] = SFFS(data, w_size)
    % Store data size
    [N, M] = size(data);

    % Initialize window size and the best performance (ideally infinity)
    bestPerf = 1000000;

    % Initialize F which contains the elements in w_size to choose among and C, which is a subset of F
    F = 0:w_size*M-1;
    C = [];

    for s=F
        V = [];
        for t=F
            if ismember(t,C) == 0
                C_tmp = [C t];

                % Arrange input and target data
                [input, target] = ArrangeData(data, C_tmp, w_size);

                [net, pNet] = NetPerf(input, target);
                V = [V; pNet t];
            end
        end
        [v, b] = min(V(:,1));
        if V(b,1) < bestPerf
            bestPerf = V(b,1);
            C = [C V(b,2)];
        else
            break
        end
    end

end

