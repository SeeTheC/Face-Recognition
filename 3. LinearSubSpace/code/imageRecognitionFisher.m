function recognitionRate = imageRecognitionFisher(projectionMatrix,subspaceDim,testCell)            
    successRate=recognise(projectionMatrix,subspaceDim,testCell);
    fprintf('**Recognition-Rate:%f \n',successRate);   
    recognitionRate=successRate;
end


function successRate = recognise(projectionMatrix,subspaceDim,testCell)    
    [row,noOfClass]=size(projectionMatrix);
    testMatrix=testCell{1};
    n=size(testCell{1},2);    
    %Label Fetch    
    testLabel=testCell{2};  
    correctRecognition=0;
    betaProjection=zeros(subspaceDim,noOfClass);     
    for i=1:n
        for c=1:noOfClass
            projection=projectionMatrix{c};
            betaProjection(:,c)=projection'*testMatrix(:,i);
        end
        betaProjectionSq=betaProjection.^2;
        betaProjectionNorm=sum(betaProjectionSq)';
        [error,predictedClass]=min(betaProjectionNorm);        
        if(predictedClass==testLabel(i))
            correctRecognition=correctRecognition+1;
        end
    end
    successRate=correctRecognition/n;
end
