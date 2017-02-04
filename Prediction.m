%predicting the stock
load('u');
load('sigma');
load('init');
load('transition');
load('Testing');
load('alpha');
Closing = zeros(1,1799);
for i = 1:5,
    Closing(i,:) = FinalClosingMatrix(i,:);
end

length = 1799;
num_State = 50;
%P(q_t = i|x_{1:t})
post = zeros(length,num_State,5);
for i = 1:5,
    for j = 1:length,
        for k = 1:num_State,
            post(j,k,i) = alpha(j,k,i)/sum(alpha(j,:,i));
        end
    end
end

% expected x_{t+1}
Predict = zeros(5,1799);
Error = zeros(5,1799);
for i = 1:5,
    Predict(i,1) = Closing(i,1);
  for j = 2:length,
      inner_Sum = 0;
      for k = 1:num_State,
          inner_Sum = inner_Sum + u(k) * transition(:,k);
      end
      Predict(i,j) = sum(post(j-1,:,i).*inner_Sum');
      Error(i,j) = Predict(i,j) - Closing(i,j);
  end
end

