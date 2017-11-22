
%% Initialization  Face Datase
% Read data will return train and  test cell. Each cell contains data and
% its associated label.
tic
downSample=0.5;
attDirpath='../../dataset/att_faces';
yaleDirpath='../../dataset/CroppedYale';
%[attrTrainImgCell,attrTestImgCell]=readData(attDirpath,'att_faces',downSample);
[yaleTrainImgCell,yaleTestImgCell,imgHeight,imgWidth]=readData(yaleDirpath,'yale',downSample);

%yaleTrainImgCell{1}=yaleTrainImgCell{1}./255;
%yaleTestImgCell{1}=yaleTestImgCell{1}./255;

toc
fprintf('**Reading of images Done.\n');
%%
trainImgCell=yaleTrainImgCell;
testImgCell=yaleTestImgCell;
totalTrainSamples=size(trainImgCell{1},2);noOfClass=max(trainImgCell{2});

%%
%% 1. Finding the EignFace : Yale Dateset

tic
[globalMean,Wpca,meanDeviatedImg]=eigenFaceUsingSVD(trainImgCell{1});
%[mean,eface,meanDeviatedImg]=eigenFace(trainImgCell{1});

toc
fprintf('**Finding Wpca.Done.\n');

%% 2. Fisher LDA
tic
[Wopt,classSpecificMean]=FisherFaceVarProjection(trainImgCell,Wpca);
toc
fprintf('**Finding Fisher Faces.Done.\n');
%% 3. Fisher Faces
tic
figure('name',strcat('Fisher Face:',int2str(i)));    
row=imgHeight;col=imgWidth;
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
%% 4. Test Data 
tic
recognitionRate=imageRecognitionFisher(Wopt,globalMean,classSpecificMean,{meanDeviatedImg,trainImgCell{2}},testImgCell);
fprintf('Recognising Test data.Done.\n');
toc