%% NPJ修订
% by 橙子
% 2022.06.02
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%% 读取数据
Data=xlsread('heilongjiang.xlsx');
%% 不同模型设置对应的自变量
input=Data(:,2:71);
output=Data(:,1);

%% 数据集切分
rng('default') % 为了固定切分
c = cvpartition(output,Holdout=0.25);
trainingIndices = training(c); % 训练集索引
testIndices = test(c); % 测试集索引
XTrain = input(trainingIndices,:);
YTrain = output(trainingIndices);
XTest = input(testIndices,:);
YTest = output(testIndices);

%% 元启发式求解
%初始化模型参数
SearchAgents_no=2; % 种群数量
% [~,dim]=size(input);
Feature_N=size(input,2);
dim=22+5.*Feature_N;
lb= zeros(1,dim);     % 自变量下界
ub= ones(1,dim);      % 自变量上界
% SVM模型
lb(6)=0.001;
ub(6)=1000;
lb(7)=0.001;
ub(7)=1000;
% 决策树
lb(8)=1;
ub(8)=100;
% KNN
lb(9)=1;
ub(9)=50;
% 神经网络
lb(10)=1;
ub(10)=300;
lb(11)=1e-08;
ub(11)=100;
% 朴素贝叶斯

Max_iteration=20; % 最大迭代次数
fobj=@(x) OBJ(x,XTrain,YTrain); %设置目标函数
[Best_pos,Best_score,TSO_curve]=ICHIO(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);



%% 汇总结果
[~,~,yfit1] = BST(XTrain,YTrain,XTrain,Best_pos);
[~,~,yfit2] = BST(XTrain,YTrain,XTest,Best_pos);
[~,i1]=max(yfit1,[],2);
[~,i2]=max(yfit2,[],2);
pred=i1-1;
pred2=i2-1;



%训练集情况
C = confusionmat(YTrain,pred) ;% 先计算训练集混淆矩阵
stats1 = statsOfMeasure(C);%训练集指标

%测试集情况
B = confusionmat(YTest,pred2) ;% 计算测试集混淆矩阵
stats2 = statsOfMeasure(B);%测试集指标

%绘制训练集ROC曲线
[x1,y1,~,auc1] = perfcurve(YTrain,yfit1(:,2),1);
figure(1)
plot(x1,y1,'r','linewidth',1.5)


%绘制测试集ROC曲线
[x2,y2,~,auc2] = perfcurve(YTest,yfit2(:,2),1);
figure(2)
plot(x2,y2,'r','linewidth',1.5)

% % 另外一种方法绘制ROC曲线  
rocObj1 = rocmetrics(YTrain,yfit1(:,2),1);

rocObj2 = rocmetrics(YTest,yfit2(:,2),1);
figure(3)
plot(rocObj1,ShowConfidenceIntervals=true)
hold on
plot(rocObj2,ShowConfidenceIntervals=true)

curveObj = plot(rocObj1,ClassNames='1', ...
    YAxisMetric="PositivePredictiveValue",XAxisMetric="TruePositiveRate");
xyData = rmmissing([curveObj.XData curveObj.YData]);
auc = trapz(xyData(:,1),xyData(:,2));
legend(join(["1" " (AUC = " string(auc) ")"],""), ...
    Location="southwest")
title("Precision-Recall Curve")

% cm = confusionchart(YTrain,pred) %混淆矩阵
% bm = confusionchart(YTest,pred2) %混淆矩阵
