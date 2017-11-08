

%% Face recognition using Eigen Faces
% We have used PCA algorithm to find eigen faces and for optimization we
% have use L matrix where L = A'A
%% Initialization Att Face Datase
% Reading the att_faces and yale database
% Read data will return train and  test cell. Each cell contains data and
% its associated label.
tic
attDirpath='../../dataset/att_faces';
yaleDirpath='../../dataset/CroppedYale';
%[attrTrainImgCell,attrTestImgCell]=readData(attDirpath,'att_faces');
[yaleTrainImgCell,yaleTestImgCell]=readData(yaleDirpath,'yale');

%yaleTrainImgCell{1}=yaleTrainImgCell{1}./255;
%yaleTestImgCell{1}=yaleTestImgCell{1}./255;

toc
fprintf('**Reading of images Done.\n');
%%
%% 1. Finding the EignFace : Yale Dateset
% Size of train data set size is 40*38 images and test data size is
% 20*38 images.
% Here we are finding eigen faces of Yale Dataset. EigenFace is calculated using SVD. It returns following : 
% 
% * mean vector
% * normalized eigen faces
% * deviated train set from its mean (Xi-X_mean)
% 

tic
trainImgCell=yaleTrainImgCell;
[globalMean,Wpca,meanDeviatedImg]=eigenFaceUsingSVD(trainImgCell{1});
%[mean,eface,meanDeviatedImg]=eigenFace(trainImgCell{1});
toc
fprintf('**Finding Wpca.Done.\n');
%% 2. Projecting Samples on lower dimensional space
tic

trainImgCell=yaleTrainImgCell;testImgCell=yaleTestImgCell;
totalTrainSamples=size(trainImgCell{1},2);noOfClass=max(trainImgCell{2});
Wpca=Wpca(:,1:totalTrainSamples-noOfClass);

%Projecting Train Samples
projectedTrainedImg=Wpca'*meanDeviatedImg;
projectedTrainedImgCell={projectedTrainedImg,trainImgCell{2}};

%Projecting Test Samples
projectedTestImg=Wpca'*bsxfun(@minus, testImgCell{1}, globalMean);
projectedTestImgCell={projectedTestImg,testImgCell{2}};

toc
fprintf('**Projecting sample to new dimensional space.Done.\n');
%% 3. Fisher LDA
tic
trainImgCell=projectedTrainedImgCell;
[Wopt,classSpecificMean]=FisherFace(trainImgCell,yaleTrainImgCell,Wpca);


toc
fprintf('**Finding Fisher Faces.Done.\n');
%% Fisher Faces
tic
figure('name',strcat('Fisher Face:',int2str(i)));    
row=192;col=168;
row=floor(row/2);col=floor(col/2);
for i=1:16
    subplot(4,4,i);
    testEigFace=Wopt(:,i);
    eigFaceImage = reshape(testEigFace,row,col);    
    colormap(jet);
    imagesc(eigFaceImage);
    title(strcat('\fontsize{10}{\color{magenta}Fisher Face: ',int2str(i),'}'));
    colorbar;
end
toc;
%% 4. Find Wopt and Model Train Samples

tic


trainImgCell=yaleTrainImgCell;
testImgCell=yaleTestImgCell;

recognitionRate=imageRecognitionFisher(Wopt,globalMean,classSpecificMean,{meanDeviatedImg,trainImgCell{2}},testImgCell);
fprintf('Recognising Test data.Done.\n');
toc

%% Recognition Plot
% Drawing Plot
figure('name','Recognition Plot: Attr Face DataSet');
x=recognitionRate{1};
y=recognitionRate{2};
plot(x,y,'--gs',...
    'LineWidth',2,...
    'MarkerSize',10,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor',[0.5,0.5,0.5]);
title('\fontsize{12}{\color{magenta}Recognition Plot: Yale DataSet}');






%%





%% Finding the EignFace : Yale Dateset
% Size of train data set size is 40*38 images and test data size is
% 20*38 images.
% Here we are finding eigen faces of Yale Dataset. EigenFace is calculated using SVD. It returns following : 
% 
% * mean vector
% * normalized eigen faces
% * deviated train set from its mean (Xi-X_mean)
% 

tic
trainImgCell=yaleTrainImgCell;
testImgCell=yaleTestImgCell;
[xMean,efaceNormalized,devTrainSet]=eigenFace(trainImgCell{1});
fprintf('Finding Eigen Faces.Done.\n');
toc

%% Testing The Probe Image : Yale Dateset
% Image Recognition function takes following parameters 
% 
% * normalized eigen face,
% * mean vector of images
% * deviated train set from its mean and associated train set label
% * test images
% * set of k largest eigen values
%
% Image recognition returns regonition rate i.e. how well test set is recognized w.r.t k


tic
ks=[1, 2, 3, 5, 10, 15, 20, 30, 50, 60, 65, 75, 100, 200, 300, 500, 1000];
recognitionRate=imageRecognition(efaceNormalized,xMean,{devTrainSet,trainImgCell{2}},testImgCell,ks);
fprintf('Recognising Test data.Done.\n');
toc

%% Recognition Plot: Yale Dateset
% Drawing Plot
figure('name','Recognition Plot: Attr Face DataSet');
x=recognitionRate{1};
y=recognitionRate{2};
plot(x,y,'--gs',...
    'LineWidth',2,...
    'MarkerSize',10,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor',[0.5,0.5,0.5]);
title('\fontsize{12}{\color{magenta}Recognition Plot: Yale DataSet}');
