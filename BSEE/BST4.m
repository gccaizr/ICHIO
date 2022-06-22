function [WW,N_all,yfit] = BST4(p_train,t_train,Xtest,x,N_first,N_last)
%% 设置固定自变量对应关系

%% 1.超参数对应关系10个
% 4个逐步回归超参数
param1=x(1);
param1=prabow(param1);
param2=x(2);
param2=prabow(param2);
param3=x(3);
param3=prabow(param3);
param4=x(4);
param4=prabow(param4);
% 6个非线性模型超参数
% 回归树最小叶大小param5(1~50)
param5=x(5);
param5=round(param5);
% SVM，可优化超参数2个
% （1）核尺度：param6（0.001~1000）
% （2）alpha系数的框约束：param7（0.001~1000）
param6=x(6);
param7=(7);
% GRP优化超参数1个：param8（sigma）高斯过程模型的噪声标准差的初始值(0.0001~1000)
param8=x(8);
% BP优化的超参数2个
% （1）隐含层节点数(整数)：param9(1~300)
% （2）正则化强度（Lambda）：param10(1e-08~100) 
param9=x(9);
param9=round(param9);
param10=x(10);
%% 2.基学习器权重4个
weight(1)=x(11);
weight(2)=x(12);
weight(3)=x(13);
weight(4)=x(14);
%% 3.特征尺度2*4=8个
Feature_scale_1=x(15);
Feature_scale_2=x(16);
Feature_scale_3=x(17);
Feature_scale_4=x(18);
Feature_scale_5=x(19);
Feature_scale_6=x(20);
Feature_scale_7=x(21);
Feature_scale_8=x(22);
%% 4.特征标记
treex=x(23:22+N_first);
svmx=x(23+N_first:22+2*N_first);
grpx=x(23+2*N_first:22+3*N_first);
bpx=x(23+3*N_first:22+4*N_first);
lmx1=x(23+4*N_first:22+4*N_first+N_last);
lmx2=x(23+4*N_first+N_last:22+4*N_first+2*N_last);
lmx3=x(23+4*N_first+2*N_last:22+4*N_first+3*N_last);
lmx4=x(23+4*N_first+3*N_last:22+4*N_first+4*N_last);
%% 5.数据集对应关系
% bootstrap自助采样
rng(10,'twister')   % 固定一个随机数种子，保证每次基学习器对应固定数据集
DATA=[t_train p_train];
Y1 = datasample(DATA,100,1,'Replace',true); % 基学习器1数据集
Y2 = datasample(DATA,100,1,'Replace',true); % 基学习器2数据集 
Y3 = datasample(DATA,100,1,'Replace',true); % 基学习器3数据集
Y4 = datasample(DATA,100,1,'Replace',true); % 基学习器4数据集

% 区分数据集
% p1=p_train(:,1:N_first);              % 前序模型特征（训练）
% p2=p_train(:,end-N_last+1:end);       % 后续模型特征(训练)
test1=Xtest(:,1:N_first);               % 前序模型特征（验证）
test2=Xtest(:,end-N_last+1:end);        % 后续模型特征（验证）
n=size(test1,1);                        % 验证集样本数
m=size(p_train,2);                      % 原始数据集特征数

% 训练集
t1=Y1(:,1);
t2=Y2(:,1);
t3=Y3(:,1);
t4=Y4(:,1);

data1=Y1(:,2:1+N_first);
data1(:,treex<Feature_scale_1)=[];   % 筛选变量

data2=Y1(:,m-N_last+2:m+1);
data2(:,lmx1<Feature_scale_2)=[];    % 筛选变量

data3=Y2(:,2:1+N_first);
data3(:,svmx<Feature_scale_3)=[];    % 筛选变量

data4=Y2(:,m-N_last+2:m+1);
data4(:,lmx2<Feature_scale_4)=[];    % 筛选变量

data5=Y3(:,2:1+N_first);
data5(:,grpx<Feature_scale_5)=[];    % 筛选变量

data6=Y3(:,m-N_last+2:m+1);
data6(:,lmx3<Feature_scale_6)=[];    % 筛选变量

data7=Y4(:,2:1+N_first);
data7(:,bpx<Feature_scale_7)=[];    % 筛选变量

data8=Y4(:,m-N_last+2:m+1);
data8(:,lmx4<Feature_scale_8)=[];    % 筛选变量

% 验证集
test_1=test1;
test_1(:,treex<Feature_scale_1)=[];   % 筛选变量

test_2=test2;
test_2(:,lmx1<Feature_scale_2)=[];    % 筛选变量

test_3=test1;
test_3(:,svmx<Feature_scale_3)=[];    % 筛选变量

test_4=test2;
test_4(:,lmx2<Feature_scale_4)=[];    % 筛选变量

test_5=test1;
test_5(:,grpx<Feature_scale_5)=[];    % 筛选变量

test_6=test2;
test_6(:,lmx3<Feature_scale_6)=[];    % 筛选变量

test_7=test1;
test_7(:,bpx<Feature_scale_7)=[];    % 筛选变量

test_8=test2;
test_8(:,lmx4<Feature_scale_8)=[];    % 筛选变量

preMD=zeros(n,4);
%% 设置一个多线程并发
parfor (i=1:4)
    if i==1
        %% 1：回归树+线性回归的残差修正模型
        %如果输入变量为空集，此基学习器无效weight=0
        if isempty(data1)       % 如果此时输入tree的自变量为空集
            weight(i)=0;         % 则线性回归的各项结果皆为0（权重、预测结果）
            preMD(:,i)=zeros(n,1);  % 此基学习训练直接结束
        else
            mdtree = fitrtree(data1,t1, 'MinLeafSize', param5);   % 回归树训练
            pretree=predict(mdtree,data1);
            pretree_test=predict(mdtree,test_1);
            % 计算残差
            err1=t1-pretree;
            % 训练线性回归模型
            if isempty(data2)        %如果此时输入线性回归的自变量为空集
                prelm1=zeros(n,1);     %模型不训练，值为0
            else
                mdlm1=stepwiselm(data2,err1, 'Criterion',param1,'Upper', 'linear','NSteps',30,'Verbose',0);
                prelm1=predict( mdlm1,test_2);
            end
            preMD(:,i)=pretree_test+prelm1;
        end

    elseif i==2
        %% 2.支持向量机+线性回归的残差修正模型
        %如果输入变量为空集，此基学习器无效weight=0
        if isempty(data3)       %如果此时输入tree的自变量为空集
            weight(i)=0;         %则线性回归的各项结果皆为0（权重、预测结果）
            preMD(:,i)=zeros(n,1);    %此基学习训练直接结束
        else
            mdSVM = fitrsvm(...
                data3, ...
                t2, ...
                'KernelFunction', 'gaussian', ...
                'KernelScale', param6, ...
                'BoxConstraint', param7, ...
                'Standardize', true);                %svm模型训练
            presvm=predict(mdSVM,data3);
            presvm_test=predict(mdSVM,test_3);
            % 计算残差
            err2=t2-presvm;
            % 训练线性回归模型
            if isempty(data4)          %如果此时输入线性回归的自变量为空集
                prelm2=zeros(n,1);     %模型不训练，值为0
            else
                mdlm2=stepwiselm(data4,err2, 'Criterion',param2,'Upper', 'linear','NSteps',30,'Verbose',0);
                prelm2=predict( mdlm2,test_4);
            end
             preMD(:,i)=presvm_test+prelm2;
        end
    elseif i==3
        %% 3.高斯过程回归模型（GPR）+线性回归的残差修正模型
        if isempty(data5)       % 如果此时输入的自变量为空集
            weight(i)=0;         % 则线性回归的各项结果皆为0（权重、预测结果）
             preMD(:,i)=zeros(n,1);  % 此基学习训练直接结束
        else
            warning off             % 关闭报警信息
            mdGRP = fitrgp(...
                data5, ...
                t3, ...
                'BasisFunction', 'constant', ...
                'KernelFunction', 'ardrationalquadratic', ...
                'Sigma', param8, ...
                'Standardize', true);                %GRP训练模型
            preGRP=predict(mdGRP,data5);
            preGRP_test=predict(mdGRP,test_5);
            % 计算残差
            err3=t3-preGRP;
            % 训练线性回归模型
            if isempty(data6)        %如果此时输入线性回归的自变量为空集
                prelm3=zeros(n,1);     %模型不训练，值为0
            else
                mdlm3=stepwiselm(data6,err3, 'Criterion',param3,'Upper', 'linear','NSteps',30,'Verbose',0);
                prelm3=predict( mdlm3,test_6);
            end
             preMD(:,i)=preGRP_test+prelm3;
        end
    else
        %% 4.BP神经网络+线性回归的残差修正模型
        %如果输入变量为空集，此基学习器无效weight=0
        if isempty(data7)       %如果此时输入的自变量为空集
            weight(i)=0;         %则线性回归的各项结果皆为0（权重、预测结果）
             preMD(:,i)=zeros(n,1);    %此基学习训练直接结束
        else
            mdBP = fitrnet(data7,t4, ...
                'LayerSizes', param9, ...
                'Activations', 'tanh', ...
                'Lambda', param10, ...
                'IterationLimit', 1000, ...
                'Standardize', true);
            preBP=predict(mdBP,data7);
            preBP_test=predict(mdBP,test_7);
            % 计算残差
            err4=t4-preBP;
            % 训练线性回归模型
            if isempty(data8)        %如果此时输入线性回归的自变量为空集
                prelm4=zeros(n,1);     %模型不训练，值为0
            else
                mdlm4=stepwiselm(data8,err4, 'Criterion',param4,'Upper', 'linear','NSteps',30,'Verbose',0);
                prelm4=predict( mdlm4,test_8);
            end
            preMD(:,i)=preBP_test+prelm4;
        end
    end
end


weight_1=weight(1);
weight_2=weight(2);
weight_3=weight(3);
weight_4=weight(4);
preMD1=preMD(:,1);
preMD2=preMD(:,2);
preMD3=preMD(:,3);
preMD4=preMD(:,4);

W=weight_1+weight_2+weight_3+weight_4+1e-08;
weight_tree=weight_1/W;
weight_svm=weight_2/W;
weight_GRP=weight_3/W;
weight_BP=weight_4/W;




%% 加权集成
yfit=weight_tree.*preMD1+...
     weight_svm.*preMD2+...
     weight_GRP.*preMD3+...
     weight_BP.*preMD4;

WW.tree=weight_tree;
WW.svm=weight_svm;
WW.GRP=weight_GRP;
WW.BP= weight_BP;

N1=size(data1,2);
N2=size(data2,2);
N3=size(data3,2);
N4=size(data4,2);
N5=size(data5,2);
N6=size(data6,2);
N7=size(data7,2);
N8=size(data8,2);

N_all=N1+N2+N3+N4+N5+N6+N7+N8;

end