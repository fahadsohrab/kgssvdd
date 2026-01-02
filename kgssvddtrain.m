function ssvddmodel = kgssvddtrain(Traindata, params, varargin)
%ssvddtrain() is a function for training a model based on "Graph Embedded Subspace Support
%Vector Data Description"
% Input
%    Traindata = Contains training data from a single (target) class for training a model.
%   'maxIter' :Maximim iteraions, Default=100
%   'C'       :Value of hyperparameter C, Default=0.1
%   'd'       :Data in lower dimension, make sure that input d<D, Default=1,
%   'eta'     :Used as step size for gradient, Default=0.1
%   'opt'     :Selection of optimisation type, Default=3 (Spectral regression based)
%              other options: 1=Gradient Based Solution, 2=Generalized eigen value based
%   'laptype' :Selection for laplacian type, 1 for PCA, 2 for S_w, 3 for knn, 4 for S_b
%   'L'       :User's defined Laplacian matrix
%   's'       :Hyperparameter for the kernel.
%   'kcluster':Number of clusters (S_w,S_b), Number of K-neighbors(knn),Default=5
%   'max'     :Input 1 for maximisation, (Default=0, minimization)
%
% Output      :gessvdd.modelparam = Trained model (for every iteration)
%             :gessvdd.Q= Projection matrix (after every iteration)
%             :gessvdd.npt=non-linear train data information, used for testing data
%Example
%essvddmodel=gessvddtrain(Traindata,'C',0.12,'d',2,'opt',2,'laptype',4);

default_params.variant = 'KGpca';
default_params.solution = 'gradient';
default_params.minimize = true;
default_params.sigma = 10;
default_params.eta = 0.01;
default_params.dim = 2;
default_params.Cval = 0.5;
default_params.maxIter = 5;
default_params.K = 5; %For kNN and k-means
default_params.consType =0;
default_params.bta =0;


given = fieldnames(params);
defaults = fieldnames(default_params);
missingIdx = find(~ismember(defaults, given));
% Assign missing fields to params
for i = 1:length(missingIdx)
    params.(defaults{missingIdx(i)}) = default_params.(defaults{missingIdx(i)});
end

%Define L for non-basic variant
if ~strcmp(params.variant, 'basic')
    if isfield(params, 'L')
        Lt = L;
    else
        Lt = laplacianselect(Traindata, params);
    end
end

[D,N] = size(Traindata);

%Compute these here to compute them only once
if strcmp(params.variant, 'basic')
    St = eye(D);
    St_inv = eye(D);
elseif ismember(params.variant, {'KGpca', 'KGkNN','KGSw','KGSb'})
    St=Lt';
else
    St = Traindata*Lt*Traindata';
end


if strcmp(params.solution, 'eig')
    St_inv = pinv(St);
elseif strcmp(params.solution, 'spectral_regression')
    Lt_inv = pinv(Lt);
end



Q = initialize_Q( D, params.dim, Traindata);

for ii=1:params.maxIter

    %% SS = sqrtm(pinv(Q*St*Q') + (10^-6)*eye(size(params.dim))); % e inverse sqrroot
    SS = pinv(real(sqrtm(Q*St*Q' + 10^-6*eye(size(Q,1))))); % e inverse sqrroot
    reducedData=SS*Q*Traindata;
    Model = svmtrain(ones(N,1), reducedData', ['-s ',num2str(5),' -t 0 -c ',num2str(params.Cval)]);
    Qiter{ii}=SS*Q;
    Modeliter{ii}=Model;

    Alphavector=fetchalpha(Model,N);
    La = diag(Alphavector) - Alphavector*Alphavector';

    %Update Q
    switch params.solution
        case 'gradient'
            Q = gradient_update(Q, La,St,Traindata,Alphavector, params);
        case 'eig'
            Sa = Traindata*La*Traindata';
            Q = eig_update(St_inv*Sa, params);
        case 'spectral_regression'
            Q = spectral_regression_update(Lt_inv*La, params, Traindata);
        case 'newton_method'
            Q = newton_update(Q,La,Traindata,Alphavector,params);
    end

    if isempty(Q)
        break;
    end

    %orthogonalize and normalize Q
    Q = OandN_Q(Q);
end

ssvddmodel.modelparam = Modeliter;
ssvddmodel.Q = Qiter;

end