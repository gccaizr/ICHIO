%% NPJ修订
% by 橙子
% 2022.06.06
% Email:1578999723@qq.com
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%% 读取数据
Data=xlsread('heilongjiang.xlsx');
%% 不同模型设置对应的自变量
input=Data(:,2:71);
output=Data(:,1);
num=size(output,1);
%% 数据集切分
rng('default') % 为了固定切分
% rng(3)
c = cvpartition(num,Holdout=0.20);
trainingIndices = training(c); % 训练集索引
testIndices = test(c); % 测试集索引
XTrain = input(trainingIndices,:);
YTrain = output(trainingIndices);
XTest = input(testIndices,:);
YTest = output(testIndices);

%% 进化求解
% 初始化模型参数
SearchAgents_no=20; % 种群数量
N_first=40;               % 前序模型特征数量
N_last=30;                % 后续模型特征数量
Feature_N=size(input,2);  % 特征总数量
Max_iteration=5; % 最大迭代次数
dim=22+4*N_first+4*N_last;
lb= zeros(1,dim);         % 自变量下界
ub= ones(1,dim);          % 自变量上界
%% 超参数范围
lb(5)=1;
ub(5)=50;
lb(6)=0.001;
ub(6)=1000;
lb(7)=0.001;
ub(7)=1000;
lb(8)=0.0001;
ub(8)=1000;
lb(9)=1;
ub(9)=300;
lb(10)=1e-08;
ub(10)=100;

fobj=@(x) OBJ4(x,XTrain,YTrain,N_first,N_last); %设置目标函数
[Best_pos,Best_score,TSO_curve]=ICHIO(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);



%收敛曲线;
semilogx(TSO_curve,'Color','r','linewidth',1.5);
title('Objective space')
xlabel('Iteration');
ylabel('Best score obtained so far');
axis tight
grid off
box on



% 寻优结果返回结果
  [~,~,T_sim2]=BST4(XTrain,YTrain,XTest,Best_pos,N_first,N_last);
  T_test=YTest;
  [~,~,T_sim1]=BST4(XTrain,YTrain,XTrain,Best_pos,N_first,N_last);
  T_train=YTrain;
  
T_sim2=round(T_sim2);
T_sim1=round(T_sim1);
%% V. 评价指标
%%  均方根误差 RMSE
M = size(YTrain,1);
N = size(YTest,1);
error1 = sqrt(sum((T_sim1 - T_train).^2)./M);
error2 = sqrt(sum((T_test - T_sim2).^2)./N);

%% 决定系数
R1 = 1 - (sum((T_sim1 - T_train).^2) / sum((T_sim1 - mean(T_train)).^2));
R2 = 1 - (sum((T_sim2 - T_test).^2) / sum((T_sim2 - mean(T_test)).^2));

% R1=rsquare(T_train,T_sim1);
% R2=rsquare(T_test,T_sim2);
%% 均方误差 MSE
mse1 = sum((T_sim1 - T_train).^2)./M;
mse2 = sum((T_sim2 - T_test).^2)./N;

%% 平均绝对误差MAE
MAE1 = mean(abs(T_train - T_sim1));
MAE2 = mean(abs(T_test - T_sim2));
%% 平均绝对百分比误差MAPE
MAPE1 = mean(abs((T_train - T_sim1)./T_train));
MAPE2 = mean(abs((T_test - T_sim2)./T_test));


%%  训练集绘图
figure
plot(1:M,T_train,1:M,T_sim1,'LineWidth',1.5)
legend('真实值','预测值')
xlabel('预测样本')
ylabel('预测结果')
string={'训练集预测结果对比';['(R^2 =' num2str(R1) ' RMSE= ' num2str(error1) ' MSE= ' num2str(mse1)  ')' ]};
title(string)
%% 预测集绘图
figure
plot(1:N,T_test,1:N,T_sim2,'LineWidth',1.5)
legend('真实值','预测值')
xlabel('预测样本')
ylabel('预测结果')
string={'测试集预测结果对比';['(R^2 =' num2str(R2) ' RMSE= ' num2str(error2)  ' MSE= ' num2str(mse2)  ')']};
title(string)

%% 打印出评价指标
disp('-----------------------训练集误差计算--------------------------')
disp('训练集的评价结果如下所示：')
disp(['平均绝对误差MAE为：',num2str(MAE1)])
disp(['均方误差MSE为：       ',num2str(mse1)])
disp(['均方根误差RMSEP为：  ',num2str(error1)])
disp(['决定系数R^2为：  ',num2str(R1)])
% disp(['平均绝对百分比误差MAPE为：  ',num2str(MAPE1)])
disp('-----------------------测试集误差计算--------------------------')
disp('测试集的评价结果如下所示：')
disp(['平均绝对误差MAE为：',num2str(MAE2)])
disp(['均方误差MSE为：       ',num2str(mse2)])
disp(['均方根误差RMSEP为：  ',num2str(error2)])
disp(['决定系数R^2为：  ',num2str(R2)])
% disp(['平均绝对百分比误差MAPE为：  ',num2str(MAPE2)])


