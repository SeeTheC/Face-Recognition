function [Wopt,classSpecificMean] = FisherFaceSampleProjection(trainImgCell,Wpca)
%   Fisher Faces
    % Initialization
    % Assuming the train set is sorted by the class label
    imgMatrix=trainImgCell{1};
    imgLabel=trainImgCell{2};
    totalTrainSamples=size(trainImgCell{1},2);
    
    [vectorSize,noOfImages]=size(imgMatrix);    
    % Finding Different Classes 
    classes= unique(imgLabel);
    noOfClass=numel(classes);
    
    %----------------------------------------------------
    
    % Taking only N-C Vectors of Wpca Projection
    %Wpca=Wpca(:,1:totalTrainSamples-noOfClass);
    %Projecting Train Samples
    %projectedTrainedImg=Wpca'*imgMatrix;    
    %vectorSize=totalTrainSamples-noOfClass;
    projectedTrainedImg=imgMatrix;
    
    % Finding global mu (mean
    globalmean=mean(projectedTrainedImg,2);
        
    % Finding per class mean
    classSpecificMean=zeros(vectorSize,noOfClass);
    datapointPerClass=zeros(noOfClass,1);
    for c=1:noOfClass                
        ci=(imgLabel==c);
        datapointPerClass(c)=sum(ci);        
        datapointIndex=find(ci,datapointPerClass(c),'first');
        classSpecificMean(:,c)=mean(projectedTrainedImg(:,datapointIndex),2);
    end
      
    
    % Finding within-class scatter (Sw) i.e
    % Sw = sigma{i=1:c} sigma{j=1:ni} (xk-mu_i)*(xk-mu_i)T
    meanDedicatedXi=zeros(vectorSize,noOfImages);    
    for i=1:noOfImages         
        meanDedicatedXi(:,i)=projectedTrainedImg(:,i)-classSpecificMean(:,imgLabel(i));
        %%Sw= Sw + (diff*diff'));
    end
    Sw=meanDedicatedXi*meanDedicatedXi';
    
    % Finding between-class scatter (Sb) i.e 
    % Sb = sigma{k=1:c} Nk * (mu_k-mu)*(mu_k-mu)T
    Sb=zeros(vectorSize,vectorSize);
    for c=1:noOfClass 
        ni=datapointPerClass(c);
        diff=classSpecificMean(:,c)-globalmean;
        Sb= Sb + (ni * (diff*diff'));
    end      
    
    [Wfld,D]=eig(Sb,Sw);
    [D,idx] = sort(diag(D),'descend');
    Wfld = Wfld(:,idx);
    
    %Taking Top C-1 Eign Vectors because after all eigen values are zero
    Wfld=Wfld(:,1:noOfClass-1); 
    %Wopt=Wpca*Wfld;
    Wopt=Wfld;
end

