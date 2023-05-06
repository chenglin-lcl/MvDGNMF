function [U, V] = random_init_data(X, nClass)
% Input
% X : Data matrix  (M*N)
% nClass : The number of clusters
% Output
% U : Clustering center matrix (M*nClass)
% V : Cluster indicator matrix (nClass*N)

[m, n] = size(X);
U = abs(rand(m, nClass));
V = abs(rand(nClass, n));

end

