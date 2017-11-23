
attDirpath='../../dataset/att_faces';
yaleDirpath='../../dataset/CroppedYale';

extendedYale='dataset/ExtendedYaleB';

[attrTrainImgCell,attrTestImgCell]=readData(attDirpath,'att_faces',1);
[yaleTrainImgCell,yaleTestImgCell]=readData(yaleDirpath,'yale',1);
% downSample=0.25;
% [ExtyaleTrainImgCell,ExtyaleTestImgCell]=readData(extendedYale,'ExtYale',downSample);
% fprintf('Reading of images Done.\n');


%% 1. Attr_Face DataSet

%% Finding the correlation : Attr_Face DataSet
% Size of train data set size is 6*32(192) images and test data size is
% 4*32(128) images.
% Here we are finding correlation matrix. It returns following : 
% 
% * mean vector
% * deviated train set from its mean (Xi-X_mean)
% 

tic
trainImgCell=attrTrainImgCell;
testImgCell=attrTestImgCell;
[devTrainSet]=correlation(trainImgCell{1});
fprintf('Finding Eigen Faces.Done.\n');
toc

%% Testing The Probe Image : Attr_Face DataSet

tic
recognitionRate=imageRecognition({devTrainSet,trainImgCell{2}},testImgCell);
fprintf('Recognising Test data.Done.\n');
toc

%%
%% 2. Yale DataSet

%% Finding the correlation : Attr_Face DataSet
tic
trainImgCell=yaleTrainImgCell;
testImgCell=yaleTestImgCell;
[devTrainSet]=correlation(trainImgCell{1});
fprintf('Finding Eigen Faces.Done.\n');
toc

%% Testing The Probe Image : Attr_Face DataSet

tic
recognitionRate=imageRecognition({devTrainSet,trainImgCell{2}},testImgCell);
fprintf('Recognising Test data.Done.\n');
toc


