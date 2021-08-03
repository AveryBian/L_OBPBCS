% clear all;	% Clear all the variables, functions, and MEX files of the workspace
% close all;	% Close all the figures windows
clc;	% Clear the contents of the command window without affecting all variables in the work environment

K = 16;	% The sparsity of feature vector 
B = 2;  % The block length of OB-PBCS algorithm 
L = 8;  % The number of scattered features in MS data

l_max = 100;  % The number of iterations of the algorithm
t_max = 50;
tau_w = 0.01;
tau_d = 0.001;
Rough_ratio = 0.5;
Refine_ratio = 0.5;
option = statset('MaxIter',Inf); % for SVM, a positive integer specifying the maximum number of iterations allowed. 

times = 1000;	% The number of trails  
SNR = 30; % Set the indensity of noise
test_num = 20;	% The number of test data

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Read data and labels%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Organize the original data

data0 = load('data_x0.mat');
data1 = load('data_x1.mat');
x_0 = data0.x;	% Select training data labeled "-1"
x_1 = data1.data_x;	% Select training data labeled "+1"       
y_0 = -ones(size(x_0,1),1);	% Markup labels "-1"
y_1 = ones(size(x_1,1),1);	% Markup labels "+1"
x_data = [x_0;x_1];  % MS data
y_data = [y_0;y_1];   % labels of MS data




% Randomly arrange data
[n,d] = size(x_data);             
index_case = randperm(n);            
x = x_data(index_case,:); 
y = y_data(index_case); 

% Add Guassian noise
rand_data = randn(size(x,1),size(x,2));
V_noise_in = norm(x)/10^(SNR/20)/norm(rand_data);
noise_in = (V_noise_in)*rand_data;
x = x+noise_in;

% Add burst noise
noise_in=zeros(size(x,1),size(x,2));
fn=0.001;
index_noise=fix((d-1)*rand(size(x,1),fix(fn*size(x,2))))+1; %Generate the positions of burst noise
for i=1:n
   noise_in(i,index_noise(i,:))=norm(mean(x))/sqrt(length(mean(x)));%Burst noise
end
x_in = x+noise_in; 

% Preprocrssing
% Data normalization
for i = 1:n
   x(i,:) = x(i,:)/norm(x(i,:),1);
end

% Gauss smoothing
sigma = 1;
for k = -d : d
    G_sigma(k+d+1) = exp(-k^2/(2*sigma^2))/(sqrt(2*pi*sigma*sigma));
end
for i=1:n
    x(i,:) = conv(x(i,:),G_sigma,'same');
end

% Data standardization
x = zscore(x);	
	   
% Set the number of training samples M or the sparsity of the feature vector K for iterations
start = 20;	
final = 90;  
gap = 10;	

Pbiht_error = zeros((final-start)/gap + 1,1);
j = 0;
       
for var = start:gap:final	  
    M = var      % M is the number of training data 
    % K = var      % K is the sparsity of the feature vector
    j = j+1;
    error_Pbiht = zeros(times,1);
    
    for ii = 1:times   
		data_x = x;
        data_y = y;

		% Randomly select test data and labels
        index_m = randperm(n);  
        x_test = data_x(index_m(1:test_num),:);
        y_test = data_y(index_m(1:test_num));	

        % Randomly select M training data and labels (no test data is included)
        data_x(index_m(1:test_num),:) = [];
        data_y(index_m(1:test_num),:) = [];
        index_train = randperm(size(data_x,1));
        x_train = data_x(index_train(1:M),:);	
        y_train = data_y(index_train(1:M)); 
        
        % Use L-OBPBCS algorithm to reconstruct feature selection vector w
       [w] = L_OBPBCS(x_train,y_train,tau_w,tau_d,K,B,L,l_max,t_max,Rough_ratio,Refine_ratio);	

        
        % Extract features according to w
        index_test_0 = find(w==0);                
        x_train(:,index_test_0) = 0;   
        x_test(:,index_test_0) = 0;	
		
        % Classification 
        svmstruct = fitcsvm(x_train,y_train,'KernelScale','auto','Standardize',true,...
            'OutlierFraction',0.05);	%Training SVM classifier    
        y_Pbiht_test = predict(svmstruct,x_test);	%Classify test data
        r_Pbiht = nnz(y_Pbiht_test - y_test);
        error_Pbiht(ii) = error_Pbiht(ii) + r_Pbiht;	%Record the number of classification error     
    end
    Pbiht_error(j) = sum(error_Pbiht)/(test_num*times);	%Calculate the classification error rate     
 end
Pbiht_accura = 1-Pbiht_error;

% Draw picture
accurateCompare = figure(1)
M = start:gap:final;
plot(M,Pbiht_accura,'-sk');
legend('OBCS, SNR=30dB');  
grid on;
xlabel('Number of training samples');
ylabel('Average accuracy rate');

timenow = datestr(now,30);
figname_accurate = ['OBCS',timenow,'.fig'];
saveas(accurateCompare,figname_accurate);	% Save figure
filename = ['OBCS','letter',timenow,'.mat'];	
save (filename);	% Save ".mat" file