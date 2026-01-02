function [eval,Predictlabel] = kgssvddtest(Testdata,Testlabel,ssvddmodel, varargin)
%ssvddtest() is a function for testing a model based on "Subspace Support
%Vector Data Description"
% Input
%   Testdata  = Contains testing data from
%   Testlabels= contains original test lables
%   ssvddmodel= contains the output obtained from "ssvddmodel=ssvddtrain(Traindata,varargin)"
% Output
%   output argument #1 = predicted labels
%   output argument #2 = accuracy
%   output argument #3 = sensitivity (True Positive Rate)
%   output argument #4 = specificity (True Negative Rate)
%   output argument #5 = precision
%   output argument #6 = F-Measure
%   output argument #7 = Geometric mean i.e, sqrt(tp_rate*tn_rate)
%Example
%[predicted_labels,accuracy,sensitivity,specificity]=gessvddtest(Testdata,testlabels,ssvddmodel);


%Iter check for fetching model and corresponding Q
iter_index = double(isempty(varargin));
if(iter_index==1)
    testiter=size(ssvddmodel.Q,2);
else
    testiter=varargin{1};
end

Q = ssvddmodel.Q;
if size(Q,2) < testiter
    Predictlabel = [];
else
    model = ssvddmodel.modelparam{testiter};
    RedTestdata=Q{testiter}* Testdata;
    Predictlabel = svmpredict(Testlabel, RedTestdata', model);
end
eval = evaluate_prediction(Testlabel,Predictlabel);

