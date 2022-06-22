function crit=prabow(param1)

if param1<=0.2
    crit='sse';
elseif param1<=0.4
    crit='aic';
elseif param1<=0.6
    crit='bic';
elseif param1<=0.8
    crit='rsquared';
else
    crit='adjrsquared';
end
end