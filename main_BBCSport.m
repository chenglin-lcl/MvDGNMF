clear;
clc;

addpath('tool/', 'dataset/', 'result/', 'user/');

load BBCSport.mat
X{1} = data{1}; %m*n
X{2} = data{2}; 
truth = double(truelabel{1}'); % n*1
class_num = length(unique(truth)); % 类别个数
maxiter = 1000;

replic = 10;
fileID = fopen('result/BBCSport_MvDGNMF_tune.txt','a');

% 参数设置
% lambda的取值是[0.001, 0.01, 0.1, 1]
% beta的取值[0.1, 0.5, 1.5, 10]
% layer的取值[50, c], [100, c]
layer_idx = 1;
lambda_value = 0.01;
beta = 0.5;

% 参数确认
layer_value = cell(1, 2);
layer_value{1} = [50, class_num];
layer_value{2} = [100, class_num];
layers = layer_value{layer_idx};
lambda = zeros(1, length(layers))*lambda_value; % 可以分别给每个试图输入不同的参数

% 指标
AC_ = zeros(1, replic);
NMI_ = zeros(1, replic);
purity_ = zeros(1, replic);
Fscore_ = zeros(1, replic);
Precision_ = zeros(1, replic);
Recall_ = zeros(1, replic);
AR_ = zeros(1, replic);

% 记录矩阵
res_record = zeros(7, replic); % 1.clustering result 2.objective function value 3.V_star
V_star_record = cell(1, replic);

for i = 1: replic
    [V_star, obj, cnt] = MvDGNMF_update(X, layers, lambda, beta, 'maxiter', maxiter);
    idx = litekmeans(V_star', class_num, 'Replicates', 20);
    result = EvaluationMetrics(truth, idx);
    AC_(i) = result(1);
    NMI_(i) = result(2);
    purity_(i) = result(3);
    Fscore_(i) = result(4);
    Precision_(i) = result(5);
    Recall_(i) = result(6);
    AR_(i) = result(7);
    res_record(:, i) = result;
    V_star_record{i} = V_star;
end
% 求每个指标均值和方差
AC(1) = mean(AC_); AC(2) = std(AC_);
NMI(1) = mean(NMI_); NMI(2) = std(NMI_);
purity(1) = mean(purity_); purity(2) = std(purity_);
Fscore(1) = mean(Fscore_); Fscore(2) = std(Fscore_);
Precision(1) = mean(Precision_); Precision(2) = std(Precision_);
Recall(1) = mean(Recall_); Recall(2) = std(Recall_);
AR(1) = mean(AR_); AR(2) = std(AR_);
% 打印结果
fprintf(fileID, "layer_num = %d, lambda_value = %g, beta = %g, error_cnt = %d\n", layer_idx, lambda_value, beta, cnt);
fprintf(fileID, "AC = %5.4f + %5.4f, NMI = %5.4f + %5.4f, purity = %5.4f + %5.4f\nFscore = %5.4f + %5.4f, Precision = %5.4f + %5.4f, Recall = %5.4f + %5.4f, AR = %5.4f + %5.4f\n",...
    AC(1), AC(2), NMI(1), NMI(2), purity(1), purity(2), Fscore(1), Fscore(2), Precision(1), Precision(2), Recall(1), Recall(2), AR(1), AR(2));
fprintf(fileID,'*****************************************************************************************************\n');
fprintf('cnt = %d\n', cnt);
plot(obj);
% 保存记录矩阵
save_file_name = ['./result/BBCSport_', num2str(int32(AC(1)*10000)), '.mat'];
save(save_file_name, 'res_record', 'obj', 'V_star_record', 'layer_idx', 'lambda_value', 'beta');



