function model = TestFisher( X, y, num_components )

    model = fisherfaces(X, y, num_components);  

end

function model = fisherfaces(X, y, num_components)
  
  N = size(X,2);
  c = max(y);
  
  % set the num_components
  if(nargin < 3)
    num_components=c-1;
  end
  
  num_components = min(c-1, num_components);
  
  % reduce dim(X) to (N-c) (see paper [BHK1997])
  Pca = pca(X, (N-c));
  Lda = lda(project(X, Pca.W, Pca.mu), y, num_components);
  
  % build model
  model.name = 'lda';
  model.mu = repmat(0, size(X,1), 1);
  model.D = Lda.D;
  model.W = Pca.W*Lda.W;
  model.P = model.W'*X;
  model.num_components = Lda.num_components;
  model.y = y;
end

function model = pca(X, num_components) 
  if(nargin < 2)
    num_components=size(X,2)-1;
  end
  % center data
  mu = mean(X,2);
  X = X - repmat(mu, 1, size(X,2));
  % svd on centered data == pca
  [E,D,V] = svd(X ,'econ');
  % build model
  model.name = 'pca';
  model.D = diag(D).^2;
  model.D = model.D(1:num_components);
  model.W = E(:,1:num_components);
  model.num_components = num_components;
  model.mu = mu;
end

function Y = project(X, W, mu)	
	X = X - repmat(mu, 1, size(X,2));
	Y = W'*X;
end

function model = lda(X, y, num_components)
 
  dim = size(X,1);
  c = max(y); 
  
  if(nargin < 3)
    num_components = c - 1;
  end
  
  num_components = min(c-1,num_components);
  
  meanTotal = mean(X,2);
  
  Sw = zeros(dim, dim);
  Sb = zeros(dim, dim);
  for i=1:c
    Xi = X(:,find(y==i));
    meanClass = mean(Xi,2);
    % center data
    Xi = Xi - repmat(meanClass, 1, size(Xi,2));
    % calculate within-class scatter
    Sw = Sw + Xi*Xi';
    % calculate between-class scatter
    Sb = Sb + size(Xi,2)*(meanClass-meanTotal)*(meanClass-meanTotal)';
  end

  % solve the eigenvalue problem
  [V, D] = eig(Sb,Sw);
  
  % sort eigenvectors descending by eigenvalue
  [D,idx] = sort(diag(D), 1, 'descend');
  
  V = V(:,idx);
  % build model
  model.name = 'lda';
  model.num_components = num_components;
  model.D = D(1:num_components);
  model.W = V(:,1:num_components);
end