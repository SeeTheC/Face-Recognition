function recognitionRate = imageRecognition(devTrainCell,testCell) 
    %devTestSet=bsxfun(@minus, testCell{1}, xMean);
    devTestSet=correlation(testCell{1});
    successRate=recognise(devTrainCell,{devTestSet,testCell{2}});
    fprintf('Recognition-Rate:%f \n',successRate);
    recognitionRate=successRate;
end


function successRate = recognise(devTrainCell,devTestCell)
    trainLabel=devTrainCell{2};
    testLabel=devTestCell{2};
    test=devTestCell{1};
    train=devTrainCell{1};
    [testrow,testcol]=size(test);
    [trainrow,traincol]=size(train);
    max=0;
    correctRecognition=0;
    for i=1:testcol
        for j=1:traincol
            C=corrcoef(train(:,j)',test(:,i)');
            
            if(C(2,1)>max)
                label=trainLabel(j);
                max=C(2,1);
            end
        end
        if(label==testLabel(i))
             correctRecognition=correctRecognition+1;
        end
        %fprintf('Recognition-Rate:%f \n',correctRecognition);
    end
    successRate=correctRecognition/testcol;
end
