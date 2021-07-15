 % This function estimate free energy by using the replica method and the
% classical Metropolis–Hastings algorithm for the linear model with Markov prior  
% (discrete-space Markov source)- Figs. 7,8, and 9.
%  Arix https://arxiv.org/abs/2009.13370.

clear
close all
clc
alpha=0.2;
etavector=0.01:0.01:0.96;
L=length(etavector);
denobeta1=zeros(1,L);
denobeta2=zeros(1,L);
for k=1:L
eta=etavector(k);
ftempA=@(z)(1-((1-((1-alpha)/alpha).*exp(-2*eta*z))./(1+((1-alpha)/alpha).*exp(-2*eta*z))).^2).*((1/sqrt(2*pi))*(1-alpha)*sqrt(eta).*exp(-((z+1).^2)*eta/2)+(1/sqrt(2*pi))*alpha*sqrt(eta).*exp(-((z-1).^2)*eta/2));
ftempB=@(z)(1-((exp(2*eta*z)-(1-alpha)/alpha)./(exp(2*eta*z)+(1-alpha)/alpha)).^2).*((1/sqrt(2*pi))*(1-alpha)*sqrt(eta).*exp(-((z+1).^2)*eta/2)+(1/sqrt(2*pi))*alpha*sqrt(eta).*exp(-((z-1).^2)*eta/2));
denobeta1(k)=integral(ftempA,0,Inf)+ integral(ftempB,-Inf,0);
end %for
delta=0.5;
alpha=1-delta;
L=length(etavector);
betasample=zeros(1,L);
MMSE=zeros(1,L);
for k=1:L
eta=etavector(k);
ftempA=@(z)(1-((1-((1-alpha)/alpha).*exp(-2*eta*z))./(1+((1-alpha)/alpha).*exp(-2*eta*z))).^2).*((1/sqrt(2*pi))*(1-alpha)*sqrt(eta).*exp(-((z+1).^2)*eta/2)+(1/sqrt(2*pi))*alpha*sqrt(eta).*exp(-((z-1).^2)*eta/2));
ftempB=@(z)(1-((exp(2*eta*z)-(1-alpha)/alpha)./(exp(2*eta*z)+(1-alpha)/alpha)).^2).*((1/sqrt(2*pi))*(1-alpha)*sqrt(eta).*exp(-((z+1).^2)*eta/2)+(1/sqrt(2*pi))*alpha*sqrt(eta).*exp(-((z-1).^2)*eta/2));
denobeta2(k)=integral(ftempA,0,Inf)+ integral(ftempB,-Inf,0);
end %for
alpha=0.2;
delta=0.5;
for k=1:L
eta=etavector(k);
MMSE(k)=(delta/(alpha+delta))*denobeta1(k)+(alpha/(alpha+delta))*denobeta2(k);
betasample(k)=(1/eta-1)/MMSE(k);
end 
F1=zeros(1,L);
for k=1:L
eta=etavector(k);
beta=betasample(k);
% Estimate free energy as a function of eta and alpha (beta fixed)
Temp1=(1/(2*beta))*((eta-1)*log2(exp(1))-log2(eta))-(1/2)*log2(2*pi/eta)-(1/2)*log2(exp(1))+(1/(2*beta))*log2(2*pi)+(1/(2*beta))*log2(exp(1));
g=@(z)((1-alpha)*sqrt(eta/(2*pi)).*exp(-(eta/2)*(z+1).^2)+ alpha*sqrt(eta/(2*pi)).*exp(-(eta/2)*(z-1).^2)).*mylog2(((1-alpha)*sqrt(eta/(2*pi)).*exp(-(eta/2)*((z+1).^2))+ alpha*sqrt(eta/(2*pi)).*exp(-(eta/2)*((z-1).^2)))); 
Temp2=integral(g,-Inf,Inf);
F1(k)=Temp1-Temp2;
end %for
F2=zeros(1,L);
for k=1:L
eta=etavector(k);
beta=betasample(k);
% Estimate free energy as a function of eta and alpha (beta fixed)
Temp1=(1/(2*beta))*((eta-1)*log2(exp(1))-log2(eta))-(1/2)*log2(2*pi/eta)-(1/2)*log2(exp(1))+(1/(2*beta))*log2(2*pi)+(1/(2*beta))*log2(exp(1));
g=@(z)(delta*sqrt(eta/(2*pi)).*exp(-(eta/2)*(z+1).^2)+ (1-delta)*sqrt(eta/(2*pi)).*exp(-(eta/2)*(z-1).^2)).*mylog2(((1-alpha)*sqrt(eta/(2*pi)).*exp(-(eta/2)*((z+1).^2))+ alpha*sqrt(eta/(2*pi)).*exp(-(eta/2)*((z-1).^2)))); 
Temp2=integral(g,-inf,inf);
F2(k)=Temp1-Temp2;
end %for
F=zeros(1,L);
for k=1:L
F(k)=(delta/(alpha+delta))*F1(k)+(alpha/(alpha+delta))*F2(k);
end

n=50;  %50
T=100;   %100
Fenergy=zeros(L,1);
for rep=1:L
beta=betasample(rep);
m=ceil(n/beta);
Phi_0=(1/sqrt(m))*randn(m,n); %initial
y_0=ones(m,1);
%Initialize
x_0=[Phi_0(:)' y_0']';  %inital state
t=1;
xcur=x_0; 
d=length(x_0);
xnext=zeros(d,1);
xstore=zeros(d,T);
xstore(:,1)=x_0;
 while t<=T
 x=xcur;
 [outpre1,out1]=P(x,m,n,alpha,delta);
 for i=1:d
   xnext(i,1) = normrnd(x(i,1),1);
 end
 [outpre2,out2]=P(xnext,m,n,alpha,delta);
 %xprime=mvnrnd(xprev,eye(d),1)'; 
 % Estimate acceptance probability 
 %P(x,m,n,alpha,delta)
 A=min(1,out2/out1);
 % Accept and Reject
 U=rand;
  if U<=A
    xcur=xnext;
  else
    xcur=x;
  end %if
 t=t+1;
 xstore(:,t)=xcur;
 end %while
% Monte-Carlo part, and estimating posterior distribution vector
 Posterdis=zeros(T,1);
 for t=1:T
 [outpre,out]=P(xstore(:,t),m,n,alpha,delta);
 Posterdis(t,1)= outpre; % P_{Y|Phi}
 end
Fenergy(rep,1)=-(1/n)*mean(log2(Posterdis(ceil(T/2):T)));
end %for
figure(1)
plot(betasample,F,'b-.',betasample,Fenergy,'-r','linewidth',2);
grid on
legend('Free Energy','Emprical Free Energy (MCMC)')
xlabel('\beta','fontsize',13)
ylabel('bits/symbol')
axis([0 5 0 8])


function [outpre,out]=P(x,m,n,alpha,delta)
% Divide x into [Phi, y] again
y=x(m*n+1:m*n+m,1);
Phi=reshape(x(1:m*n,1),[m,n]);
maxiter=100;
t=1;
F=zeros(maxiter,1);
while t<=maxiter %MC to estimate q_{Y|Phi}
% Generate a Markov Chain u^n
u=zeros(n,1);
u(1,1)=binornd(1,alpha/(alpha+delta));
for i=2:n
 if (u(i-1,1)==0) 
     u(i,1)=binornd(1,alpha);
 else
     u(i,1)=binornd(1,1-delta);
 end
end
for i=1:n
    if u(i,1)==0
      u(i,1)=-1;
    end
end
% Estimate q_{Y|Phi}
L1=((1/(2*pi))^(m/2))*exp(-(norm(y-Phi*u))^2/2); % P_{Y|Phi,X}
F(t)=L1; 
t=t+1;
end %while
outpre=mean(F); % P_{Y|Phi}
%out=((m/(2*pi))^(m*n/2))*exp(-(m/2)*(norm(Phi,'fro'))^2)*outpre; % P_{Phi Y}
out=exp((m/2)*(n*log(m/(2*pi))-(norm(Phi,'fro'))^2))*outpre;   % P_{Phi Y}
end
function out = mylog2(in) %This function set xlog(x) to 0 as x to 0.
  out = log2(in);
  out(~in) = 0;
end






