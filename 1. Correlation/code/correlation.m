function [allImageDeviation]=correlation(imgMatrix)
    % Computing allImageMean = (1/P)*sum(imgMatrix j's)    (j = 1 : P)
    allImageMean=mean(imgMatrix); % allImageMean=M*N x 1
    allImageStd=std(imgMatrix);
    %allImageMean=norm(imgMatrix);
    % Computing the deviation  X=M*N x P
    allImageDeviation=bsxfun(@minus, imgMatrix', allImageMean');
    allImageDeviation=bsxfun(@times, allImageDeviation, 1./(allImageStd'));
    allImageDeviation=allImageDeviation';        
end