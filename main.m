%% 元启发算法的特征筛选
% by 橙子
% 2021.12.19
% Email:1578999723@qq.com
clc,clear
data=readmatrix('heilongjiang.xlsx');
input=zscore(data(3:end,4:end));%特征
output=data(3:end,3);%标签
L=length(output);  %总样本个数
num=310;         %设置训练集个数
train_x = input(1:num,:);
train_y = output(1:num,:);
test_x = input(num+1:end,:);
test_y = output(num+1:end,:);
%% 元启发式求解
%初始化模型参数
SearchAgents_no=30; % 种群数量
[~,dim]=size(train_x);
dim=dim+3;%多设置3个维度
lb= [zeros(1,dim-2),0.01,0.01];      % 自变量下界
ub= [ones(1,dim-2),100,100];      % 自变量上界
Max_iteration=100; % 最大迭代次数
fobj=@(x) ClusteringCost(x,train_x,train_y); %设置目标函数
[Best_pos,Best_score,TSO_curve]=ICHIO(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%收敛曲线;
semilogx(TSO_curve,'Color','r','linewidth',1.5);
title('Objective space')
xlabel('Iteration');
ylabel('Best score obtained so far');
axis tight
grid on
box on
legend('ICHIO') 

%% 汇总结果
%自变量数量i
b=[];
for j=1:70
    if  Best_pos(j)<Best_pos(71)
        Best_pos(j)=0;
        b=[b,j];
    else
        Best_pos(j)=1;
    end
end
num=sum(sum(Best_pos>0));
train_x(:,b)=[];
mdl=fitcsvm(train_x,train_y, ...
    'KernelFunction','RBF',...
    'KernelScale',Best_pos(72) , ...
    'BoxConstraint', Best_pos(73), ...
    'ClassNames', [1; 0]);
%% 结果汇总
test_x(:,b)=[];
pred=predict(mdl,train_x);
pred2=predict(mdl,test_x);
%训练集情况
C = confusionmat(train_y,pred) ;% 先计算训练集混淆矩阵
stats = statsOfMeasure(C)%训练集指标
% cm = confusionchart(train_y,pred)
%外部测试集情况
B = confusionmat(test_y,pred2) ;% 计算测试集混淆矩阵
stats = statsOfMeasure(B)%测试集指标
% bm = confusionchart(test_y,pred2)
%绘制ROC曲线
[x1,y1,~,auc1] = perfcurve(test_y,pred2,1); 
plot(x1,y1,'r','linewidth',1)

