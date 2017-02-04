%Training-Baum Welch algorithm
load('Stock closing.mat')

%closing value data
num_Stock = 5;
Closing = zeros(num_Stock,1799);
for i = 1:num_Stock,
    Closing(i,:) = FinalClosingMatrix(i,:);
end

%number of states 
num_States = 50;

%initial guess probability
init = rand(1,num_States);
init = init/sum(init(:));
u = rand(1,num_States) * mean(Closing(:));
sigma = rand(1,num_States) * var(Closing(:));
transition = rand(num_States,num_States);
for i = 1:num_States,
    transition(i,:) = transition(i,:)/(sum(transition(i,:)));
end
num_Iteration = 20;
likelihood = zeros(1,num_Iteration);

for n = 1:num_Iteration,
    %forward probablities 
    length = size(Closing,2);
    alpha = zeros(length,num_States,num_Stock);
    beta = zeros(length,num_States,num_Stock);

    %scaling factors
    S_alpha = zeros(length,1,num_Stock);
    for i = 1:num_Stock,
        Sequence = Closing(i,:);
        for j = 1:length,
                if(j==1),
                    for k = 1:num_States,
                        alpha(j,k,i) = init(k) * prob_Gaussian(Sequence(1),u(k),sigma(k));
                    end
                else
                    for k = 1:num_States,
                        alpha(j,k,i) = prob_Gaussian(Sequence(j),u(k),sigma(k)) * sum(alpha(j-1,:,i).*...
                            transition(:,k)');
                    end
                end
                S_alpha(j,1,i) = 1/(max(alpha(j,:,i)));
                alpha(j,:,i) = alpha(j,:,i) * S_alpha(j,1,i);
        end
    end

    %backward probabilities 

    S_beta = zeros(length,1,num_Stock);
    for i = 1:num_Stock,
        Sequence = Closing(i,:);
            for j = length:-1:1,
                if(j==length),
                    for k = 1:num_States,
                        beta(j,k,i) = 1.0;
                    end
                else
                    for k = 1:num_States,
                        beta(j,k,i) = 0.0;
                        for q = 1:num_States,
                            beta(j,k,i) = beta(j,k,i) + beta(j+1,q,i)*prob_Gaussian(Sequence(j+1),u(q)...
                                ,sigma(q)) * transition(k,q); 
                        end
                    end
                end
                S_beta(j,1,i) = 1.0/max(beta(j,:,i));
                beta(j,:,i) = S_beta(j,1,i) * beta(j,:,i);
            end
    end

    gamma = zeros(length,num_States,num_Stock);
    for i = 1:num_Stock,
        for j = 1:length,
            for k = 1:num_States,
                gamma(j,k,i) = alpha(j,k,i) * beta(j,k,i) / (sum(alpha(j,:,i).*beta(j,:,i)));
            end
        end
    end

    gamma_2 = zeros(num_States,num_States,length-1,num_Stock);
    for i = 1:num_Stock,
        Sequence = Closing(i,:);
        for j = 1:length - 1,
            temp_sum = 0;
            for k = 1:num_States,
                for p = 1:num_States,
                    gamma_2(k,p,j,i) = alpha(j,k,i) * transition(k,p) * beta(j+1,p,i) *...
                        prob_Gaussian(Sequence(j+1),u(p),sigma(p));
                    temp_sum = temp_sum + gamma_2(k,p,j,i);                       
                end
            end
            gamma_2(:,:,j,i) = gamma_2(:,:,j,i)/temp_sum;
        end
    end

    %data loglikelihood
    L = 0;
    for i = 1:num_Stock,
        L = L + log(sum(alpha(length,:,i))) - sum(log(S_alpha(:,1,i)));
    end

    %update parameters

    %initial probability
    for k = 1:num_States,
        init(k) = sum(gamma(1,k,:))/num_Stock;
    end

    %transition probability
    for k = 1:num_States,
        for p = 1:num_States,
            transition(k,p) = sum(sum(gamma_2(k,p,:,:)))/sum(sum(gamma(1:end-1,k,:)));
        end
    end

    %update u
    
    for k = 1:num_States,
        sum_1 = 0;
        for i = 1:num_Stock,
            sum_1 = sum_1 + (sum(gamma(:,k,i).*Closing(i,:)'));
        end
       u(k) = sum_1/sum(sum(gamma(:,k,:)));
    end
   
    %update sigma
    
    for k = 1:num_States,
        sum_1 = 0;
        for i = 1:num_Stock,
            sum_1 = sum_1 + sum(gamma(:,k,i).*((Closing(i,:)' - u(k)).^2));
        end
        sigma(k) = sum_1/sum(sum(gamma(:,k,:)));
    end
    likelihood(n) = L;
    if(n>1 && abs((likelihood(n)-likelihood(n-1))/likelihood(n-1))<0.01),
        break;
    end
end


save('init','init');
save('transition','transition');
save('u','u');
save('sigma','sigma');
save('alpha','alpha');
