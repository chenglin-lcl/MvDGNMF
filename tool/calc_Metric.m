% Run k-means n times and report means and standard deviations of the
% performance measures.
%
% -------------------------------------------------------
% Input:
%       X:  data matrix (rows are samples) 行是样本 n*m
%       k:  number of clusters
%       truth:  truth cluster indicators 行向量 1*n
%     
%
% Output:
%       AC:  clustering accuracy (mean +stdev)
%       nmi:  normalized mutual information (mean +stdev)
%       purity
%
function [AC, NMI, purity] = calc_Metric(X, k, truth, replic)

if (min(truth)==0)
    truth = truth+1;
end

% acc, nmi, Pu, Fscore, Precision, Recall, ARI

AC_ = zeros(1, replic);
NMI_ = zeros(1, replic);
purity_ = zeros(1, replic);
Fscore_ = zeros(1, replic);
Precision_ = zeros(1, replic);
Recall_ = zeros(1, replic);
AR_ = zeros(1, replic);
for i=1:replic
    idx = litekmeans(X, k, 'Replicates', 20);
    result = EvaluationMetrics(truth, idx);
    AC_(i) = result(1);
    NMI_(i) = result(2);
    purity_(i) = result(3);
    Fscore_(i) = result(4);
    Precision_(i) = result(5);
    Recall_(i) = result(6);
    AR_(i) = result(7);
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
fprintf("AC = %5.4f + %5.4f, NMI = %5.4f + %5.4f, purity = %5.4f + %5.4f\nFscore = %5.4f + %5.4f, Precision = %5.4f + %5.4f, Recall = %5.4f + %5.4f, AR = %5.4f + %5.4f\n",...
    AC(1), AC(2), NMI(1), NMI(2), purity(1), purity(2), Fscore(1), Fscore(2), Precision(1), Precision(2), Recall(1), Recall(2), AR(1), AR(2));

end