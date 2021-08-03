function [w0] = L_OBPBCS(data,y,tau_w,tau_d,K,B,L,l_max,t_max,Rough_ratio,Refine_ratio)
% data is the traininng data.
% y is the labels corresponding to the data.
% tau_w and tao_d is the step size.
% K is the sparsity of feature selection vector.
% S is the number for scattered feature selection vector.
% L is the length of block.
% l_max and t_max are the number of the iterations for obtaining w and D.
% Rough_ratio is the ratio of rough selection.
% Refine_ratio is the ratio of refining.
% w is the reconstructed feature selection vector.   

    [~,N] = size(data);
    Rough_num = Rough_ratio*N;
    Refine_num = Refine_ratio*K;
    w0 = zeros(N,1);
    w = ones(N,1);
    
   % LDA based rough selection step.
    [data_block,pos] = LDA_rough_selection (w,data,y,Rough_num,N);
    
   % OBPBCS based feature selection.
    w0_temp = OBPBCS(data_block,y,tau_w,tau_d,K,L,B,l_max,t_max);
    w0(pos) = w0_temp;
   
   % LDA based feature refining.
    w0 = LDA_refine (w0,data,y,Refine_num);
            
end


