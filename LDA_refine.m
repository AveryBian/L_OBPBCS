function [w] = LDA_refine (w,data,y,Refine_num)

% Fine the non-zero part in w and data.
se = find(w~=0);
Data_select=data(:,se);
flag1=y==1;
flag2=y==-1;
X1=Data_select(flag1,:);
X2=Data_select(flag2,:);

% LDA projection.
Mu1=mean(X1)';
Mu2=mean(X2)';
s1=cov(X1);%
s2=cov(X2);
Sw=s1+s2;
SB=(Mu1-Mu2)*(Mu1-Mu2)';
invsw=inv(Sw);
invsw_by_sb=invsw*SB;
[V,DE]=eig(invsw_by_sb);

%  Select features based on the refining ratio.
value=[abs(V(:,1)),se];
value1=sortrows(value,1);
w(value1(end-Refine_num+1:end,2))=0;