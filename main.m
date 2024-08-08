
clc
clear all
close all
tic
%% 读入数据
 load WebKB
for nn = 1:length(X)
    [N,m] = size(X{nn});
     X{nn} =mapstd(X{nn});
end

%% 数据连接
 viewcell=X;
viewmat=[];
K=size(viewcell,2);
for i=1:K
    viewcell{i}=double(viewcell{i});
    viewmat = [viewmat viewcell{i}];
end
[n,~] = size(viewmat);
s=length(unique(Y'));  % s为数据类的个数
m=s+25;
ACC=[];NMI=[];Purity=[];T=[];RESULT=[];
%% 主程序
alphalist=[1e-7];
for iter0=1:length(alphalist)
    alpha=alphalist(iter0);
for iter1=1:1                                                                                              
tic
%% k-Means
[label, cluster_centers] = litekmeans(viewmat, m);
%% Get B
sigma=10;Bcell=[];temp1=[];
for i=1:K
    temp0=size(viewcell{i},2);
    temp1=[temp1,temp0];
end
cluster_centers_cell=mat2cell(cluster_centers,m,temp1);
for ii=1:K
Bcell{ii}=ConstructA_NP(viewcell{ii}',cluster_centers_cell{ii}');
Bcell{ii}=Bcell{ii}';
end
%% 初始化
Z=zeros(m,n);
alphaV1=[];J=[];
alphaV=1/K*ones(1,K);%6
for i=1:K
    Z=Z+alphaV(i)*Bcell{i};
    S{i}=Bcell{i}*Bcell{i}';
    D{i}=diag(sum(S{i},2));
    L{i}=D{i}-S{i};
end
Z=Z';
W=rand(n,s);
V=rand(m,s);  
temp0=(Z-W*V').^2;
deta=1e0*sqrt(sum(sum(temp0)/(2*m)));

%alpha=1e-5;
%% 交互迭代
for iter=1:30
   %得到H
   temp1=(Z-W*V').^2;
   temp2=(sum(temp1,2))./(2*deta^2);
   H=diag(exp(-temp2)./(deta^2));
   %得到W,V
   SS=0;DD=0;
   for i=1:K
       SS=SS+alphaV(i)*S{i};
       DD=DD+alphaV(i)*D{i};
   end
   W=W.*(H*Z*V)./(H*W*V'*V);
   V=V.*(Z'*H*W+alpha*SS*V)./(V*W'*H*W+alpha*DD*V);
   Wtemp=sqrt(sum(W.*W,1));
   %得到alpha
   Btemp={};B0=[];L0=[];
   for i=1:K
       Btemp{i}=(sqrt(H)*Bcell{i}')';
       Btemp=reshape(Btemp{i},m*n,1);
       Ltemp=reshape(L{i},m*m,1);
       B0=[B0 Btemp];
       L0=[L0 Ltemp];
       clear Btemp;
   end
   B=B0'*B0;
   temp5=reshape(sqrt(H)*W*V',m*n,1);
   temp6=reshape(V*V',m*m,1);
   b1=2*temp5'*B0;
   b2=alpha*temp6'*L0;
   b=b1-b2;
   [alphaV, val,p] = SimplexQP_ALM(B, b, 1e-3,1.05,1);
   Z=zeros(m,n);
   LL=0;
   for i=1:K
       Z=Z+alphaV(i)*Bcell{i};
       LL=LL+alphaV(i)*L{i};
   end
   Z=Z';
   J0=trace(Z'*H*Z)-2*trace(V*W'*H*Z)+trace(V*W'*H*W*V')+alpha*trace(V'*LL*V);
   J=[J J0];
   clear B0;
end
[maxv,ind]=max(W,[],2);
[acc,nmi,purity ] = ClusteringMeasure(Y, ind);
ACC=[ACC acc];
NMI=[NMI nmi];
Purity=[Purity purity];
t=toc;
T=[T t];
result=[acc,nmi,purity,t]
RESULT=[RESULT;result];
end
end
record=[mean(RESULT(:,1)),std(RESULT(:,1));
 mean(RESULT(:,2)),std(RESULT(:,2));
 mean(RESULT(:,3)),std(RESULT(:,3));
  mean(RESULT(:,4)),std(RESULT(:,4));
 ];
record = record'

