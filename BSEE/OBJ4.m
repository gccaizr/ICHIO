function  erro=OBJ4(x,p_train,t_train,N_first,N_last)

num=size(t_train,1);
%进行K折交叉验证
rng('default') % 为了固定切分
k=10;
c = cvpartition(num,'KFold',k);
Y=[];
preY=[];
drop=0;
for i=1:k
    trainingIndices = training(c,i); % 训练集索引
    valIndices = test(c,i); % 验证集索引
    XTrain = p_train(trainingIndices,:);
    YTrain = t_train(trainingIndices);
    Xval = p_train(valIndices,:);
    Yval = t_train(valIndices);
    [WW,N_all,yfit] = BST4(XTrain,YTrain,Xval,x,N_first,N_last);
    if sum(sum(yfit))==0
        drop=1;
        break;
    end
    Y=[Y;Yval];
    preY=[preY;yfit];
end

if drop==1
    erro=1000;
else
objmse=mse(preY,Y);
weight_tree=WW.tree;
weight_svm=WW.svm;
weight_GRP=WW.GRP;
weight_BP=WW.BP;
% 正则化项1
L1=100;
Re1=L1*(abs(weight_tree).^2+abs(weight_svm).^2+abs(weight_GRP).^2+abs(weight_BP).^2);

% 正则化项2
L2=0.1;
Re2=L2*(N_all);

%% 目标函数=误差(mse)+正则化项 
erro=objmse+Re1+Re2;

end
end