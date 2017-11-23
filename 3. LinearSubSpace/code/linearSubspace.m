% Linear Supspace
% For each face, use three or more images taken under different lighting 
% directions to construct a 3D basis for the linear subspace. 
% Note that the three basis vectors have the same dimensionality as the training 
% images and can be thought of as basis images
function [ projectionsMatrix ] = linearSubspace(trainImgCell,subspaceDim)
    %Init
    imgMatrix=trainImgCell{1};
    imgLabel=trainImgCell{2};
    classes= unique(imgLabel);
    noOfClass=numel(classes); 
    projectionsMatrix=cell([1,noOfClass]);
    for c=1:noOfClass
        ci=(imgLabel==c);               
        datapointIndex=find(ci,subspaceDim,'first');
        mat=imgMatrix(:,datapointIndex);
        projectionsMatrix{c}=mat;%normalize(mat);
    end
end

function wNorm=normalize(w)
    wSq=w.^2;
    wDis=sum(wSq).^0.5;
    wNorm= bsxfun(@times, w, 1./wDis);
end

