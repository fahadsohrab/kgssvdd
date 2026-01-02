function Q = initialize_Q(D,d, Traindata)
% rng(1234)
%%%% Projection Matrix
%Q=randn(d,D); %19200);
Q = pca(Traindata')';
if size(Q,1)<d
extraD=d-size(Q,1);
Q=[Q;eps * randn(extraD,D)];
end
Q = Q(1:d,:);
[Q_Pos, R]=qr(Q',0); % create an orthogonal matrix Q for witch it holds Q*Q'=I but not Q'*Q=I
Q=Q_Pos';
%Q_init=Q;
clear Q_Pos;
clear R;
A=sqrt(sum(Q'.^2));
for i=1:length(A)
Q(i,:)=Q(i,:)/A(i);
end
end