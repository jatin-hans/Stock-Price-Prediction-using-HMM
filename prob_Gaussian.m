function [ prob ] = prob_Gaussian( x,u,sigma )
%x-value 
%u-mean, sigma-variance
    prob = (1/(sqrt(2*pi*abs(sigma)))) * exp((-1/2) * (x-u)^2 * (1/sigma));
end

