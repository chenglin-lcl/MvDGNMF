function [W, W1, L] = get_consensus_graph(S_all, view_num)
% S_all: cell array
% S_star: 
S_star = S_all{1};
if view_num > 1
    for view_idx = 2: view_num
        S_star = min(S_star, S_all{view_idx});
    end
end
W = S_star;
W1 = sparse(diag(sum(W, 2)));
L = W1 - W;
end

