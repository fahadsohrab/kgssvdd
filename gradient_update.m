function Q = gradient_update(Q,La,St,Traindata,Alphavector,params)

if strcmp(params.variant, 'basic') ||  strcmp(params.variant, 'ellipsoid')
const= generalconstraintESSVDD(params.consType,params.Cval,Q,Traindata,Alphavector);
else
  const=0; 
end

Sa = Traindata*La*Traindata';

if strcmp(params.variant, 'basic') ||  strcmp(params.variant, 'ellipsoid')
    Grad = 2*Q*Sa+(params.bta*const);
else
    Sinv=pinv(Q*St*Q'); %L is assumed symmetric
    Grad = 2*Sinv*Q*Sa - 2*Sinv*Q*Sa*Q'*Sinv*Q*St;  % the constt thing is not added yet 
end

if params.minimize, eta = params.eta; else, eta = - params.eta; end
        
Q = Q - eta*Grad; 

end