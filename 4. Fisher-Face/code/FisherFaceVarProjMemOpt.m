% Fisher Faces
% In this, simplification method is projecting Sw and Sb on Wpca. 
function [Wopt,normWopt,classSpecificMean] = FisherFaceVarProjMemOpt1(trainImgCell,Wpca)
    % Initialization
    % Assuming the train set is sorted by the class label
    imgMatrix=trainImgCell{1};
    imgLabel=trainImgCell{2};
    totalTrainSamples=size(trainImgCell{1},2);noOfClass=max(trainImgCell{2});
        
    [vectorSize,noOfImages]=size(imgMatrix);
    % Finding Different Classes 
    classes= unique(imgLabel);
    noOfClass=numel(classes);
    % Finding global mu (mean
    globalmean=mean(imgMatrix,2);
        
    % Finding per class mean
    classSpecificMean=zeros(vectorSize,noOfClass);
    datapointPerClass=zeros(noOfClass,1);
    for c=1:noOfClass                
        ci=(imgLabel==c);
        datapointPerClass(c)=sum(ci);        
        datapointIndex=find(ci,datapointPerClass(c),'first');
        classSpecificMean(:,c)=mean(imgMatrix(:,datapointIndex),2);
    end
          
    % Finding within-class scatter (Sw) i.e
    % Sw = sigma{i=1:c} sigma{j=1:ni} (xk-mu_i)*(xk-mu_i)T
    meanDedicatedXi=zeros(vectorSize,noOfImages);    
    for i=1:noOfImages         
        meanDedicatedXi(:,i)=imgMatrix(:,i)-classSpecificMean(:,imgLabel(i));
        %Sw= Sw + (diff*diff'));
    end
   
    % Finding between-class scatter (Sb) i.e 
    % Sb = sigma{k=1:c} Nk * (mu_k-mu)*(mu_k-mu)T
    bclassDiff=zeros(vectorSize,noOfClass);
    for c=1:noOfClass 
        ni=datapointPerClass(c);
        diff=classSpecificMean(:,c)-globalmean;
        %Sb= Sb + (ni * (diff*diff'));
        bclassDiff(:,c)= (sqrt(ni) * (diff));
    end
    
    % Taking only N-C Vectors of Wpca Projection
    Wpca=Wpca(:,1:totalTrainSamples-noOfClass);
    %Projecting Sb and Sw on Wpca domain
    WtSbW=(Wpca'*bclassDiff)*(bclassDiff'*Wpca);    
    WtSwW=(Wpca'*meanDedicatedXi)*(meanDedicatedXi'*Wpca);
    
    [Wfld,D]=eig(WtSbW,WtSwW);
    
    [D,idx] = sort(diag(D),'descend');
    Wfld = Wfld(:,idx);    
    
    %Taking Top C-1 Eign Vectors because after all eigen values are zero
    Wfld=Wfld(:,1:noOfClass-1); 
    Wopt=Wpca*Wfld;
    normWopt=normalize(Wopt);    
    
end

function wNorm=normalize(w)
    wSq=w.^2;
    wDis=sum(wSq).^0.5;
    wNorm= bsxfun(@times, w, 1./wDis);
end