% **************************************************************************************
%   ********************   Random Under-Sampling (RUS)          ********************
%   ********************     Created by Sana Mujeeb             ********************
%   ******************** COMSATS University Islambad Pakistan   ********************
%   ********************            sansanik@yahoo.com          ********************
% **************************************************************************************

function random_sample = RUS(imbalanced_data, Samles_No)
% Input an imbalanced data and number of samples
 
x = imbalanced_data;
y = x(:,1);
% 
r = x(  y == 1 , : );
t = x(  y == 0 , : );
if size(r,1) > size(t,1)
p = floor((size(t,1))/1.25);
else
p = floor((size(r,1))/1.25);
end
rng('default') % 为了固定切分
for i = 1 : Samles_No
    out1 = randperm(size(r,1),p);
    out1 = r(out1,:);
    out2 = randperm(size(t,1),p);
    out2 = t(out2,:);
randSamp = [out1; out2];
random_sample{i, :} = randSamp(randperm(size(randSamp, 1)), :); 
end
end

