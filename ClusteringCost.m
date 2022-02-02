function ER = ClusteringCost(x, train_x,train_y)
%处理数据集
x_1=x(1:70);
b=[];
for j=1:70
    if  x_1(j)<x(71)
        b=[b,j];
        x_1(j)=0;
    else
        x_1(j)=1;
    end
end

data_x=train_x.*x_1;
if length(b)==70
    ER=100;
else
    data_x(:,b)=[];
    mdl=fitcsvm(data_x,train_y, ...
        'KernelFunction','RBF',...
        'KernelScale',x(72) , ...
        'BoxConstraint', x(73), ...
        'ClassNames', [0; 1] );
    % 执行交叉验证
    partitionedModel = crossval(mdl, 'KFold', 10);

    % 计算错误率
    ER = kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
end

end