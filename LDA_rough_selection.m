function [data_rough,pos] = LDA_rough_selection (w,data,y,Rough_num)

% Initialization.
w0_temp = [];
data_rough = [];
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
[~,position]=sort(w);    
for count_b = 1:Rough_num            
    data_rough = [data_rough data(:,position(end-count_b+1))];
    w0_temp = [w0_temp;w(position(end-count_b+1))];
    pos = [pos position(end-count_b+1)];
end

