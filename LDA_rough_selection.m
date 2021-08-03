function [data_block,pos] = LDA_rough_selection (w,data,y,Rough_num,N)

% Initialization.
w0_temp = [];
data_block = [];
pos = [];

% Fine the non-zero part in w and data.
select_index = w~=0;
Data_select=data(:,select_index);
flag1=y==1;
flag2=y==-1;
X1=Data_select(flag1,:);
X2=Data_select(flag2,:);

% Calculate the rough selection coefficient.
Mu1=mean(X1)';
Mu2=mean(X2)';
s1=var(X1);
s2=var(X2);
for i=1:size(Data_select,2)
    value(i) = ( (Mu1(i)-Mu2(i))^2 ) / (s1(i)^2 *s2(i)^2);
end
w = value;

% Select features based on the rough selection ratio.
count_i=1;
groupn = ceil (N);
energe(groupn,1)=norm(w((groupn-1)*1+1:end),1);
for count_j=1:groupn-1
    energe(count_j,1)=norm(w(count_i:count_i),1);
    count_i=count_j+1;
end
[~,position]=sort(energe);    
for count_b = 1:Rough_num            
    data_block = [data_block data(:,position(end-count_b+1))];
    w0_temp = [w0_temp;w(position(end-count_b+1))];
    pos = [pos position(end-count_b+1)];
end

