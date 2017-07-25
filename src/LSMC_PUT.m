S = 36;
K = 40;
sig = 0.2;
r = 0.06;
T = 1.0;

n = 100000;
m = T*50;

conf = 0.99;

%% SIMULATION MC antithetic

dt = T/m;
sims = ones(n,m+1)*S;

noise = randn(n/2,m);

R_plus =  exp((r-sig^2/2)*dt+sig*sqrt(dt)*noise);
R_moins= exp((r-sig^2/2)*dt+sig*sqrt(dt)*-noise);

sims(1:n/2,2:end) = sims(1:n/2,2:end).*cumprod(R_plus,2);
sims(n/2+1:end,2:end) = sims(n/2+1:end,2:end).*cumprod(R_moins,2);

if n <= 1000
    plot((0:dt:T),sims');
end
%% Valuation for PUT
time= (0:dt:T)';
CFP= zeros(n,m+1);

CFP(:,end)= max(K-sims(:,end),0); % Fill exercise value at T

% Laguerre's polynomials
L_0 = @(x) ones(length(x),1);
L_1 = @(x) (1-x);
L_2 = @(x) 1/2*(2-4*x-x.^2);

for ii = m:-1:2
    exo= max(K-sims(:,ii),0); %value of exercise at t= ii*dt
    atmIdx= find(exo>0);%gets ATM indices
    
    xP= sims(atmIdx,ii)/S; % Gets spot price for atm paths
    yP= CFP(atmIdx,ii+1).*exp(-r*dt); %Discounted CF from continuation 
    
    %coeffs= [L_0(xP) L_1(xP) L_2(xP)]\yP;
    coeffs= regress(yP,[L_0(xP) L_1(xP) L_2(xP)]);%both lines equivalent
    
    exIdx= exo(atmIdx) > [L_0(xP) L_1(xP) L_2(xP)]*coeffs;% trouve les paths qui sont itm qui sont exercised

    CFP(atmIdx(exIdx),ii)= exo(atmIdx(exIdx));
    nIdx = setdiff((1:n)',atmIdx(exIdx));
    CFP(nIdx,ii)= CFP(nIdx,ii+1)*exp(-r*dt);
end
CFP(:,1)= CFP(:,2)*exp(-r*dt);
put = mean(CFP(:,1));
z = norminv(0.5+conf/2);
se = std(CFP(:,1));
brackets = [put-z*se/sqrt(n); put+z*se/sqrt(n)];
