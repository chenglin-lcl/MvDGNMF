function [H_star, obj, error_cnt] = MvDGNMF_update(X, layers, lambda, beta, varargin)
% Input: X(M*nSmp)
% layers >= 2

pnames = {'maxiter', 'tolfun'};
dflts  = {500, 1e-5};
[maxiter, tolfun] = internal.stats.parseArgs(pnames,dflts,varargin{:});

view_num = length(X); % 视图个数
layer_num = length(layers); % 分解层数

Z = cell(view_num, layer_num);
H = cell(view_num, layer_num);
omega = ones(1, view_num)/view_num;
obj = zeros(1, 1);

% 数据进行归一化
for i = 1: view_num
    [X{i}, ~] = data_normalization(X{i}, [], 'std');
end

% 构建最优图
W = cell(view_num, 1);
D = cell(view_num, 1);
L = cell(view_num, 1);
options = [];
options.WeightMode = 'HeatKernel';  
options.Metric = 'Euclidean';
options.NeighborMode = 'KNN';
options.k = 5;
% 对每个视图构建图
for view_idx = 1: view_num
    W{view_idx} = constructW(X{view_idx}',options);
    D{view_idx} = diag(sum(W{view_idx}, 1));
    L{view_idx} = D{view_idx} - W{view_idx};
end

% 预训练
for view_idx = 1: view_num
    for layer_idx = 1: layer_num
        if layer_idx == 1
           upper_H = X{view_idx}; 
        else
           upper_H = H{view_idx, layer_idx-1};
        end
        % random进行初始化
        [Z{view_idx, layer_idx}, H{view_idx, layer_idx}] = random_init_data(upper_H, layers(layer_idx)); 
    end    
end
H_star = abs(rand(size(H{1, layer_num})));


% 更新错误计数器
error_cnt = 0;

% 迭代更新
for iter = 1: maxiter
    for view_idx = 1: view_num
        for layer_idx = 1: layer_num
            % update H
            if layer_idx == 1
                num = (Z{view_idx, layer_idx}')*X{view_idx} + Z{view_idx, layer_idx+1}*H{view_idx, layer_idx+1} + lambda(layer_idx)*H{view_idx, layer_idx}*W{view_idx};
                den = (Z{view_idx, layer_idx}')*Z{view_idx, layer_idx}*H{view_idx, layer_idx} + H{view_idx, layer_idx} + lambda(layer_idx)*H{view_idx, layer_idx}*D{view_idx};
                H{view_idx, layer_idx} = H{view_idx, layer_idx} .* (num ./ max(den, 1e-9));
            elseif layer_idx < layer_num
                num = (Z{view_idx, layer_idx}')*H{view_idx, layer_idx-1} + Z{view_idx, layer_idx+1}*H{view_idx, layer_idx+1} + lambda(layer_idx)*H{view_idx, layer_idx}*W{view_idx};
                den = (Z{view_idx, layer_idx}')*Z{view_idx, layer_idx}*H{view_idx, layer_idx} + H{view_idx, layer_idx} + lambda(layer_idx)*H{view_idx, layer_idx}*D{view_idx};
                H{view_idx, layer_idx} = H{view_idx, layer_idx} .* (num ./ max(den, 1e-9));
            else
                num = (Z{view_idx, layer_idx}')*H{view_idx, layer_idx-1} + beta*omega(view_idx)*H_star + lambda(layer_idx)*H{view_idx, layer_idx}*W{view_idx};
                den = (Z{view_idx, layer_idx}')*Z{view_idx, layer_idx}*H{view_idx, layer_idx} + beta*omega(view_idx)*H{view_idx, layer_idx} + lambda(layer_idx)*H{view_idx, layer_idx}*D{view_idx};
                H{view_idx, layer_idx} = H{view_idx, layer_idx} .* (num ./ max(den, 1e-9));
            end
            % update U
            if layer_idx == 1
                num = X{view_idx}*(H{view_idx, layer_idx}');
                den = Z{view_idx, layer_idx}*H{view_idx, layer_idx}*(H{view_idx, layer_idx}');
                Z{view_idx, layer_idx} = Z{view_idx, layer_idx} .* (num ./ max(den, 1e-9));
            else
                num = H{view_idx, layer_idx-1}*(H{view_idx, layer_idx}');
                den = Z{view_idx, layer_idx}*H{view_idx, layer_idx}*(H{view_idx, layer_idx}');
                Z{view_idx, layer_idx} = Z{view_idx, layer_idx} .* (num ./ max(den, 1e-9));
            end
        end
    end
    temp = zeros(size(H_star));
    for view_idx = 1: view_num
        temp = temp + omega(view_idx)*H{view_idx, layer_num};
    end
    H_star = temp;
       
   obj(iter) = Calc_obj_value(X, Z, H, lambda, L, beta, layers, omega, H_star);
   fprintf('iter = %d, obj = %g\n', iter, obj(iter));
   if (iter>=2)&&(obj(iter)>obj(iter-1))
      error_cnt = error_cnt + 1; 
   end
    if (iter>2) && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<tolfun)|| iter==maxiter
        break;
    end
   
end

end

function [obj] = Calc_obj_value(X, Z, H, lambda, L, beta, layers, omega, H_star)
view_num = length(X); % 视图个数
layer_num = length(layers); % 分解层数
obj = 0;
for view_idx = 1: view_num
    for layer_idx = 1: layer_num
        if layer_idx == 1
            obj = obj + (norm(X{view_idx}-Z{view_idx, layer_idx}*H{view_idx, layer_idx}, 'fro').^2);
        else
            obj = obj + (norm(H{view_idx, layer_idx-1}-Z{view_idx, layer_idx}*H{view_idx, layer_idx}, 'fro').^2);
        end
        obj = obj + lambda(layer_idx)*trace(H{view_idx, layer_idx}*L{view_idx}*(H{view_idx, layer_idx}'));
    end
    obj = obj + beta*omega(view_idx)*(norm(H{view_idx, layer_num}-H_star, 'fro').^2);
end
end





