function [WW,N_all,yfit] = BST(p_train,t_train,Xtest,x)

%% 测试集模型对应
input_test_1=Xtest;
input_test_2=Xtest;
input_test_3=Xtest;
input_test_4=Xtest;
input_test_5=Xtest;
%% 1.欠采样的数据集与基学习器对应关系=1:5
Feature_N=size(p_train,2);                 % 特征数量
imbalanced_data=[t_train p_train];
random_sample=RUS(imbalanced_data, 5);     % 欠采样5个数据集
x_1=x(1:5);
[~,S]=sort(x_1);                           % 得到对应关系
%% 2.关于7个超参数=6:12
% SVM模型2个：6，7
% ②核尺度：（0.001~1000）;
% ①SVM的框约束（0.001~1000）;
param1=x(6);
param2=x(7);
% 决策树1个：8
% 最大分裂树:（1~(样本量-1)）
param3=x(8);
% KNN1个：9
% 领点个数：1~100
param4=x(9);
param4=round(param4);     % 整数处理
% 神经网络分类2个：10，11
% 隐含层节点数:(1~300)
% 正则化强度:(1e-08~100)
param5=x(10);
param5=round(param5);     % 整数处理
param6=x(11);
% 朴素贝叶斯分类器

% 判断核类型
if x(12)<=0.25
    param7='box';
elseif x(12)<=0.5
    param7='epanechnikov';
elseif x(12)<=0.75
    param7='normal';
else
    param7='triangle';
end
%% 3.5个模型的权重=13:17
weight_svm=x(13);
weight_tree=x(14);
weight_knn=x(15);
weight_bp=x(16);
weight_nb=x(17);
%% 4.5个特征尺度=18:22
Feature_scale_1=x(18);
Feature_scale_2=x(19);
Feature_scale_3=x(20);
Feature_scale_4=x(21);
Feature_scale_5=x(22);
%% 基学习器1=23:22+Feature_N
data1=random_sample{S(1),1};
input1=data1(:,2:end);
output1=data1(:,1);
n=size(Xtest,1);                        % 欠采样样本数
svmx=x(23:22+Feature_N);
input1(:,svmx<Feature_scale_1)=[];        % 筛选变量
input_test_1(:,svmx<Feature_scale_1)=[];  % 筛选变量
if isempty(input1)
    weight_svm=0;                   % 权重为0
    Scores1=zeros(n,2);             % 输出为0
    e1=0;
else
    mdl1=fitcsvm(input1,output1, ...
        'KernelFunction','RBF',...
        'KernelScale',param1 , ...
        'BoxConstraint', param2, ...
        'Standardize', true, ...
        'ClassNames', [0; 1] );
    [~,Scores1]=predict(mdl1,input_test_1);
    e1 = 1-loss(mdl1,input1,output1);
end

%% 基学习器2=23+Feature_N:22+2.*Feature_N
data2=random_sample{S(2),1};
input2=data2(:,2:end);
output2=data2(:,1);
treex=x(23+Feature_N:22+2*Feature_N);
input2(:,treex<Feature_scale_2)=[];  % 筛选变量
input_test_2(:,treex<Feature_scale_2)=[];  % 筛选变量
if isempty(input2)
    weight_tree=0;
    Scores2=zeros(n,2);             % 输出为0
    e2=0;
else
    mdl2 = fitctree(...
        input2, ...
        output2, ...
        'MaxNumSplits', param3, ...
        'Surrogate', 'off', ...
        'ClassNames', [0; 1]);
    [~,Scores2]=predict(mdl2,input_test_2);
    e2 = 1-loss(mdl2,input2,output2);
end

%% 基学习器3=23+2.*Feature_N:22+3.*Feature_N
data3=random_sample{S(3),1};
input3=data3(:,2:end);
output3=data3(:,1);
knnx=x(23+2*Feature_N:22+3*Feature_N);
input3(:,knnx<Feature_scale_3)=[];       % 筛选变量
input_test_3(:,knnx<Feature_scale_3)=[];       % 筛选变量
if isempty(input3)
    weight_knn=0;
    Scores3=zeros(n,2);             % 输出为0
    e3=0;
else
    mdl3 = fitcknn(...
        input3, ...
        output3, ...
        'NumNeighbors', param4, ...
        'Standardize', true, ...
        'ClassNames', [0; 1]);
    [~,Scores3]=predict(mdl3,input_test_3);
    e3=1-loss(mdl3,input3,output3);
end

%% 基学习器4=23+3.*Feature_N:22+4.*Feature_N
data4=random_sample{S(4),1};
input4=data4(:,2:end);
output4=data4(:,1);
bpx=x(23+3*Feature_N:22+4*Feature_N);
input4(:,bpx<Feature_scale_4)=[];                % 筛选变量
input_test_4(:,bpx<Feature_scale_4)=[];          % 筛选变量
if isempty(input4)
    weight_bp=0;
    Scores4=zeros(n,2);             % 输出为0
    e4=0;
else
    mdl4 = fitcnet(...
        input4, ...
        output4, ...
        'LayerSizes', param5, ...
        'Lambda', param6, ...
        'IterationLimit', 1000, ...
        'Standardize', true, ...
        'ClassNames', [0; 1]);
    [~,Scores4]=predict(mdl4,input_test_4);
    e4=1-loss(mdl4,input4,output4);
end

%% 基学习器5=23+4.*Feature_N:22+5.*Feature_N
data5=random_sample{S(5),1};
input5=data5(:,2:end);
output5=data5(:,1);
nbx=x(23+4*Feature_N:22+5*Feature_N);
input5(:,nbx<Feature_scale_5)=[];          % 筛选变量
input_test_5(:,nbx<Feature_scale_5)=[];          % 筛选变量
if isempty(input5)
    weight_nb=0;
    Scores5=zeros(n,2);                    % 输出为0
    e5=0;
else
    mdl5 = fitcnb(...
        input5, ...
        output5, ...
        'Kernel', param7, ...
        'Support', 'Unbounded', ...
        'DistributionNames', 'kernel', ...
        'ClassNames', [0; 1]);
    [~,Scores5]=predict(mdl5,input_test_5);
    e5=1-loss(mdl5,input5,output5);
end

%% 加权集成
weight_1=weight_svm*e1;
weight_2=weight_tree*e2;
weight_3=weight_knn*e3;
weight_4=weight_bp*e4;
weight_5=weight_nb*e5;

yfit=weight_1.*Scores1+...
    weight_2.*Scores2+...
    weight_3.*Scores3+...
    weight_4.*Scores4+...
    weight_5.*Scores5;

WW.svm=weight_1;
WW.tree=weight_2;
WW.knn=weight_3;
WW.bp= weight_4;
WW.nb= weight_5;

N1=size(input1,2);
N2=size(input2,2);
N3=size(input3,2);
N4=size(input4,2);
N5=size(input5,2);
N_all=N1+N2+N3+N4+N5;
% yfit=yfit./(yfit(:,1)+yfit(:,2));   %归一化

end