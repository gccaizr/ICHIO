function  erro=OBJ(x,p_train,t_train)


%进行K折交叉验证
rng('default') % 为了固定切分
k=10;
c = cvpartition(t_train,'KFold',k);
Y=[];
preY=[];
drop=0;
weight_1=zeros(1,k);
weight_2=zeros(1,k);
weight_3=zeros(1,k);
weight_4=zeros(1,k);
weight_5=zeros(1,k);
for i=1:k
    trainingIndices = training(c,i); % 训练集索引
    valIndices = test(c,i); % 验证集索引
    XTrain = p_train(trainingIndices,:);
    YTrain = t_train(trainingIndices);
    Xval = p_train(valIndices,:);
    Yval = t_train(valIndices);
    [WW,N_all,yfit] = BST(XTrain,YTrain,Xval,x);
    weight_1(i)=WW.svm;
    weight_2(i)=WW.tree;
    weight_3(i)=WW.knn;
    weight_4(i)=WW.bp;
    weight_5(i)=WW.nb;
    if sum(sum(yfit))==0
        drop=1;
        break;
    end
    Y=[Y;Yval];
    preY=[preY;yfit];
end

if drop==1
    erro=100;
else
%% 1.交叉验证得到模型AUC值
% [~,~,~,auc] = perfcurve(Y,preY(:,2),1);
[~,i1]=max(preY,[],2);
pred2=i1-1;
C = confusionmat(Y,pred2) ;% 先计算训练集混淆矩阵
stats1 = statsOfMeasure(C);%训练集指标
auc=table2array(stats1(9,3));

%% 2.正则化项模型权重+特征数量
weight_svm=mean(weight_1);
weight_tree=mean(weight_2);
weight_knn=mean(weight_3);
weight_bp=mean(weight_4);
weight_nb=mean(weight_5);

%     weight_svm=x(13);
%     weight_tree=x(14);
%     weight_knn=x(15);
%     weight_bp=x(16);
%     weight_nb=x(17);
% 
%     Feature_scale_1=x(18);
%     Feature_scale_2=x(19);
%     Feature_scale_3=x(20);
%     Feature_scale_4=x(21);
%     Feature_scale_5=x(22);
% 
%     % 基学习器1
%     Feature_N=size(p_train,2);
%     svmx=x(23:22+Feature_N);
%     input1=p_train;
%     input1(:,svmx<Feature_scale_1)=[];  % 筛选变量
%     if isempty(input1)
%         weight_svm=0;
%     end
%     N1=size(input1,2);
% 
%     % 基学习器2
%     treex=x(23+Feature_N:22+2*Feature_N);
%     input2=p_train;
%     input2(:,treex<Feature_scale_2)=[];  % 筛选变量
%     if isempty(input2)
%         weight_tree=0;
%     end
%     N2=size(input2,2);
% 
% 
%     % 基学习器3
%     knnx=x(23+2*Feature_N:22+3*Feature_N);
%     input3=p_train;
%     input3(:,knnx<Feature_scale_3)=[];       % 筛选变量
%     if isempty(input3)
%         weight_knn=0;
%     end
%     N3=size(input3,2);
% 
%     % 基学习器4
%     bpx=x(23+3*Feature_N:22+4*Feature_N);
%     input4=p_train;
%     input4(:,bpx<Feature_scale_4)=[];          % 筛选变量
%     if isempty(input4)
%         weight_bp=0;
%     end
%     N4=size(input4,2);
% 
%     % 基学习器5
%     nbx=x(23+4*Feature_N:22+5*Feature_N);
%     input5=p_train;
%     input5(:,nbx<Feature_scale_5)=[];          % 筛选变量
%     if isempty(input5)
%         weight_nb=0;
%     end
%     N5=size(input5,2);
% 
    % 正则化项1
    L1=0.01;
    Re1=L1*(abs(weight_svm).^2+abs(weight_tree).^2+abs(weight_knn).^2+ ...
        abs(weight_bp).^2+abs(weight_nb).^2);

    % 正则化项2
    L2=0.0001;
    Re2=L2*(N_all);

    %% 目标函数=误差(mse)+正则化项
    erro=(1-auc)+Re1+Re2;
end
end