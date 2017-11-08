% load function files from subfolders aswell
%addpath (genpath ('.'));

% load data
%[X y width height names] = read_images('/home/philipp/facerec/data/at');

tic
yaleDirpath='../../dataset/CroppedYale';
[yaleTrainImgCell,yaleTestImgCell]=readData(yaleDirpath,'yale');
toc
fprintf('**Reading of images Done.\n');

X=yaleTrainImgCell{1};
y=yaleTrainImgCell{2};
height=192/2;width=168/2; 
%%
% compute a model
fisherface = TestFisher(X,y,38);

%% plot fisherfaces
figure; hold on;
for i=1:min(16, size(fisherface.W,2))
  subplot(4,4,i);
  comp = reshape(fisherface.W(:,i), height,width);
  imagesc(comp);
  colormap(gray);
  title(sprintf('Fisherface #%i', i));
  colorbar;
end

%% 2D plot of projection (first three classes)
figure; hold on;
for i = findclasses(fisherface.y, [1,2,3])
  text(fisherface.P(1,i), fisherface.P(2,i), num2str(fisherface.y(i)));
end

%% 3D plot of projection (first three classes)
if(rows(fisherface.P) >= 3)
  figure; hold on;
  for i = findclasses(fisherface.y, [1,2,3])
    % LineSpec: red dots 'r.'
    plot3(fisherface.P(1,i), fisherface.P(2,i), fisherface.P(3,i), 'r.'), view(45,-45);
    text(fisherface.P(1,i), fisherface.P(2,i), fisherface.P(3,i), num2str(fisherface.y(i)));
  end
end

pause;