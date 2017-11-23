
%% Initialization Face Datase
% Read data will return train and  test cell. Each cell contains data and
% its associated label.
tic
clear
databaseType=2;
if(databaseType==1)% attr
    attDirpath='../../dataset/att_faces';
    [attrTrainImgCell,attrTestImgCell,attImgHeight,attrImgWidth,attrSubspaceDim]=readData(attDirpath,'att_faces',1);
    trainImgCell=attrTrainImgCell;
    testImgCell=attrTestImgCell;
    imgHeight= attImgHeight;imgWidth=attrImgWidth;
    subspaceDim=attrSubspaceDim;

elseif(databaseType==2)% yale
    yaleDirpath='../../dataset/CroppedYale';
    [yaleTrainImgCell,yaleTestImgCell,yaleImgHeight,yaleImgWidth,yaleSubspaceDim]=readData(yaleDirpath,'yale',1);
    trainImgCell=yaleTrainImgCell;
    testImgCell=yaleTestImgCell;
    imgHeight= yaleImgHeight;imgWidth=yaleImgWidth;
    subspaceDim=yaleSubspaceDim;
elseif(databaseType==3)% extended yale
    extendedYale= '/media/khursheed/4E20CD3920CD2933/wamp/ExtendedYaleB';
    [extendedYaleTrainImgCell,extendedYaleTestImgCell,extendedYaleImgHeight,extendedYaleImgWidth,extendedSubspaceDim]=readData(extendedYale,'extendedyale',1/5);
    trainImgCell=extendedYaleTrainImgCell;
    testImgCell=extendedYaleTestImgCell;
    imgHeight= extendedYaleImgHeight;imgWidth=extendedYaleImgWidth;    
    subspaceDim=extendedSubspaceDim;

end    

totalTrainSamples=size(trainImgCell{1},2);noOfClass=max(trainImgCell{2});

toc
fprintf('**Reading of images Done.\n');
%%
%% 1. Finding the LinearSubspace : 
tic
[ projectionMatrix ] = linearSubspace(trainImgCell,subspaceDim);
toc;
%% 2. Recognistion on Test
tic
recognitionRate = imageRecognitionFisher(projectionMatrix,subspaceDim,testImgCell); 
toc

