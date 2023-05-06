function [data, label] = data_normalization(data, label, method)

% Data normalization
% 列是样本
% Inputs:
%       data                data of size d*n, where d is dimension and n is number of sets 
%       label               label data of size 1*n, n is number of labels.
% Output:
%       data                normalized data of size dxn, where d is dimension and n is number of sets 
%       label               label data of size 1xn, n is number of labels. (no change!)
%
%
% Created by H.Kasai on July 04, 2017
    
    switch method
        case 'mean_std'
            data = data - repmat(mean(data),[size(data,1),1]);
            data = data./repmat(sqrt(sum(data.^2)),[size(data,1),1]);
        case 'mean'
            data = data - repmat(mean(data),[size(data,1),1]);
        case 'dimension_mean'
            data = data - repmat(mean(data,2),[1, size(data,2)]);
        case 'none'
            % Do nothing.
        case 'std' % normalize each column data to l2-norm
            %data = normc(data);
            data = data./repmat(sqrt(sum(data.^2)),[size(data,1),1]); %对data矩阵中的每一列向量进行单位化
            % data = [a1, a2, ..., an]; -> data = [a1/||a1||, a2/||a2||, ..., an/||an||]
        case 'sample_mean_per_class'
            classes = unique(label);
            class_num = length(classes);
            for j = 1 : class_num
                train_idx = find(label == classes(j));    
                class_mean = mean(data(:,train_idx), 2);
                data(:,train_idx) = data(:,train_idx) - repmat(class_mean, 1, length(train_idx));                     
            end 
        otherwise
            
    end  
    
end
