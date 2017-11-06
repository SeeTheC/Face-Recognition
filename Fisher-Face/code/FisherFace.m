function [ output ] = FisherFace(trainImgCell )
%   Fisher Faces
    output=0;
    % Initialization
    % Assuming the train set is sorted by the class label
    imgMatrix=trainImgCell{1};
    imgLabel=trainImgCell{2};
    % Finding Different Classes 
    classes= unique(imgLabel);
    noOfClass=numel(classes);
    % Finding global mu (mean
    globalmean=mean(imgMatrix,2);
    vectorSize=size(globalmean,1);
    
    % Finding per class mean
    classSpecificMean=zeros(vectorSize,noOfClass);
    datapointPerClass=zeros(noOfClass,1);
    for c=1:noOfClass                
        ci=(imgLabel==c);
        datapointPerClass(c)=sum(ci);        
        datapointIndex=find(ci,datapointPerClass(c),'first');
        classSpecificMean(:,c)=mean(imgMatrix(:,datapointIndex),2);
    end
    
    % Finding between-class scatter (Sb) i.e 
    % Sb = sigma{k=1:c} Nk * (mu_k-mu)*(mu_k-mu)T
    Sb=zeros(vectorSize,vectorSize);
    for c=1:noOfClass 
        ni=datapointPerClass(c);
        diff=classSpecificMean(:,c)-globalmean;
        Sb= Sb + (ni * (diff*diff'));
    end
end

