function [U, V] = KMeansdata(X, nClass)
% Input
% X : Data matrix  (M*N)
% nClass : The number of clusters
% Output
% U : Clustering center matrix (M*nClass)
% V : Cluster indicator matrix (nClass*N)
[label] = litekmeans(X', nClass, 'Replicates',20);
label_u = unique(label, 'stable');
label_res = zeros(size(label));
for i = 1: nClass
   label_res(label==label_u(i))=i; 
end

V = max(label2idcmat(label_res), 0.01);
U = X*(V');

end

function [H] = label2idcmat(label)
% input: label = [2, 1, 3, 2];
% output: H = [0, 1, 0, 0;
%              1, 0, 0, 1;
%              0, 0, 1, 0];
% label = round(label);
nSmp = length(label);
if (min(label)==0)
    label = label+1;
end
nClass = length(unique(label));
H = zeros(nClass, nSmp);
for nsmp_idx = 1: nSmp
    h = H(:, nsmp_idx);
    h(label(nsmp_idx)) = 1;
    H(:, nsmp_idx) = h';
end
end
