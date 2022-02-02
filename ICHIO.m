

function [Best_pos,Best_score,CHIO_curve]=ICHIO(SearchAgents_no,Max_iteration,lb,ub,dim,fobj) %开始优化

    if(max(size(ub)) == 1)
       ub = ub.*ones(1,dim);
       lb = lb.*ones(1,dim);  
    end
    PopSize=SearchAgents_no; 

    MaxAge = 100;
    C0 = 1; % number of solutions have corona virus
    Max_iter=Max_iteration; %/*The number of cycles for foraging {a stopping criteria}*/
    SpreadingRate = 0.05;   % Spreading rate parameter
    ObjVal = zeros(1,PopSize);
    Age = zeros(1,PopSize);
    % Initializing arrays
    swarm=zeros(PopSize,dim);

    %% 改进点：利用sobol序列改进种群初始化
    swarm=initializationNew(PopSize,dim,ub,lb);

    for i=1:PopSize
      ObjVal(i)=fobj(swarm(i,:));
    end

 [minValue ,index]=min(ObjVal);
 Best_score = minValue;
 Best_pos = swarm(index,:);

    Fitness=calculateFitness(ObjVal);

 %% update the status of the swarms (normal, confirmed) 
    %%the minmum C0 Immune rate will take 1 status which means 
    %%infected by corona 
    Status=zeros(1,PopSize);     
    for i=1:C0
        Status(fix(rand*(PopSize))+1)=1;  
    end        
      itr=0;   % Loop counter
      while itr<Max_iter
          %% 改进点：引入非线性惯性权重
          w = 1-(itr/Max_iter)^2;
          for i=1:PopSize
              NewSol=swarm(i,:);
%               NewSol_f=NewSol(1:131);
%               while  sum(sum(NewSol_f<0.2))<101
%                   NewSolConst = initialization(1,dim-1,ub,lb);
%                   swarm(i,1:131) = NewSolConst(:);
%                   Status(i) = 0;
%                   NewSol=swarm(i,:);
%               end
              CountCornoa = 0;
              % find the set of confirmed solutions
              confirmed = randperm(size(find(Status==1),2));
              confirmed1 = find(Status==1);
              %find(Status==1);
              % find the set of normal solutions
              normal = randperm(size(find(Status==0),2));
              normal1 = find(Status==0);
              % find the set of recovered solutions
              recovered = find(ObjVal & Status==2);
              [cost,Index3]=min(recovered);
              for j=1: dim
                  r = rand();  % select a number within range 0 to 1.
                 if ((r < SpreadingRate/3)&&(size(confirmed1,2)>0))
                      % select one of the confirmed solutions
                      z=round(1+(size(confirmed1,2)-1)*rand);   
                      zc= confirmed1(z);
                      % modify the curent value
                      NewSol(j) = w*swarm(i,j)+(swarm(i,j)-swarm(zc,j))*(rand-0.5)*2; %引入惯性权重
                      % manipulate range between lb and ub
                      NewSol(j)= min(max(NewSol(j),lb(j)),ub(j)); 
                      CountCornoa = CountCornoa + 1;                              
                  
                  elseif ((r < SpreadingRate/2) &&size(normal1,2)>0)
                      % select one of the normal solutions
                      z=round(1+(size(normal1,2)-1)*rand);
                      zn= normal1(z);
                      % modify the curent value
                      NewSol(j) = w*swarm(i,j)+(swarm(i,j)-swarm(zn,j))*(rand-0.5)*2;%引入惯性权重
                      % manipulate range between lb and ub
                      NewSol(j)= min(max(NewSol(j),lb(j)),ub(j)); 
                  
                  elseif (r < SpreadingRate && size(recovered,2)>0)
                      % modify the curent value
                      NewSol(j) = w*swarm(i,j)+(swarm(i,j)-swarm(Index3,j))*(rand-0.5)*2;%引入惯性权重
                      % manipulate range between lb and ub
                      NewSol(j)= min(max(NewSol(j),lb(j)),ub(j));  
                  end
              end
             
            
            
              %evaluate new solution
              ObjValSol=fobj(NewSol);
              FitnessSol=calculateFitness(ObjValSol);
              
              % Update the curent solution  & Age of the current solution
              if (ObjVal(i)>ObjValSol) 
                swarm(i,:)=NewSol;
                Fitness(i)=FitnessSol;
                ObjVal(i)=ObjValSol;
              else
                  if(Status(i)==1)
                      Age(i) = Age(i) + 1;
                  end
              end            
                           
              % change the solution from normal to confirmed
              if ((Fitness(i) < mean(Fitness))&& Status(i)==0 && CountCornoa>0)
                  Status(i) = 1;
                  Age(i)=1;
              end
              
              % change the solution from confirmed to recovered
              if ((Fitness(i) >= mean(Fitness))&& Status(i)==1)
                  Status(i) = 2; 
                  Age(i)=0;
              end
              
              % killed the current soluion and regenerated from scratch
              if(Age(i)>=MaxAge)
                  NewSolConst = initialization(1,dim,ub,lb);
                  swarm(i,:) = NewSolConst(:);
                  Status(i) = 0;
              end
              
          end
          
          itr=itr+1;
          [minValue ,index]=min(ObjVal);
          if minValue<Best_score
              Best_score = minValue;
              Best_pos = swarm(index,:);
          end
          
          %% %% 改进点：最优解逐维高斯变异
              w = 1-(i/Max_iter)^2;
              for j = 1:dim
                  Temp =Best_pos;
                  Temp(j) = w.*Temp(j) + randn()*Temp(j);
                  if Temp(j)>ub(j)
                      Temp(j)=ub(j);
                  end
                  if Temp(j)<lb(j)
                     Temp(j)=lb(j) ;
                  end
                  fTemp = fobj(Temp);
                   if fTemp < Best_score
                        X(index,:) = Temp;
                        Best_score = fTemp;
                        Best_pos = Temp;
                   end
              end
          
           CHIO_curve(itr) = Best_score; 
          Best_score=Best_score
      end
  
end
