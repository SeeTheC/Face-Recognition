function recognitionRate = imageRecognitionFisher(fisherFace,globalMean,classSpecificMean,devTrainCell,testCell)    
    %Tries to recognise the test image with K eigen Values    
    devTestSet=bsxfun(@minus, testCell{1}, globalMean);
    successRate=recognise(fisherFace,classSpecificMean,devTrainCell,{devTestSet,testCell{2}});
    fprintf('Recognition-Rate:%f \n',successRate);   
    recognitionRate=successRate;
end


function successRate = recognise(V,classSpecificMean,devTrainCell,devTestCell)
    
    % fisherCoff: number_of_img 
    VT=V';
    noOfClass=max(devTrainCell{2});
    classSpecificMean=zeros(size(devTrainCell{1},1),noOfClass);
    for c=1:noOfClass                
        ci=(devTrainCell{2}==c);
        datapointIndex=find(ci,sum(ci),'first');
        classSpecificMean(:,c)=mean(devTrainCell{1}(:,datapointIndex),2);
    end
    mat=devTrainCell{1};    
    %mat=classSpecificMean;
    
    aEigenCoff=VT*mat;
    bEigenCoff=VT*devTestCell{1};
    n=size(devTestCell{1},2);
    
    %Label Fetch
    trainLabel=devTrainCell{2};
    testLabel=devTestCell{2};  
    correctRecognition=0;
              
    for i=1:n
        bCoff_i=bEigenCoff(:,i);
        % j= || aj-b||2
        alphaMinusBeta=bsxfun(@minus, aEigenCoff, bCoff_i);
        alphaMinusBetaSq=alphaMinusBeta.^2;
        alphaMinusBetaNorm=sum(alphaMinusBetaSq)';
        % Recognisation: j= min || aj-b||2  for j in the train set       
        [error,index]=min(alphaMinusBetaNorm);        
        %fprintf('testImg_%d : %d\n',i,index);
        if(trainLabel(index)==testLabel(i))
            correctRecognition=correctRecognition+1;
        end
    end
    successRate=correctRecognition/n;
end
