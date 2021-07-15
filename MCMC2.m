% This function estimate free energy by using the replica method and the
% classical Metropolis–Hastings algorithm for the linear model with Markov prior 
% continuous-space Markov (Harris) source where S=1 in Section VI-A (Fig 10).
%  Arix https://arxiv.org/abs/2009.13370.

clear
close all
clc
etavector=0.01:0.01:0.96;
sigmanot=1;
snot=1;
L=length(etavector);
F=zeros(1,L);
denobeta=zeros(1,L);
betavector=zeros(1,L);
% Estimate free energy and capacity as a function of %beta%
for k=1:L
eta=etavector(k);
%ftemp=@(u,xnot) (1/sqrt(2*pi))*(1/sqrt(sigmanot^2/(1-nu^2)))*exp(-xnot.^2*(1-nu^2)/(2*sigmanot^2))...
 %   .*(((eta*u*sqrt(snot)+nu*xnot/sigmanot^2)/(snot*eta+1/sigmanot^2)).^2)...
  %  *(sqrt(eta)/(2*pi*sigmanot))*(sqrt(pi/(snot*eta/2+1/(2*sigmanot^2))))...
   % .*exp((eta*u*sqrt(snot)+nu*xnot/sigmanot^2).^2/(4*(snot*eta/2+1/(2*sigmanot^2)))-(eta*u.^2/2+nu^2*xnot.^2/(2*sigmanot^2)));
%denobeta(k)=snot*(sigmanot^2+nu^2*sigmanot^2/(1-nu^2))-snot*integral2(ftemp,-Inf,Inf,-Inf,Inf);
denobeta(k)=(snot*sigmanot^2/eta)/(snot*sigmanot^2+1/eta);
betavector(k)=(1/eta-1)/denobeta(k);    
end
for k=1:L
    beta=betavector(k);
    F(k)=(1/2)*log2((2*pi*exp(1))*(snot*sigmanot^2+1/etavector(k)))+(1/(2*beta))*((etavector(k)-1)*log2(exp(1))-log2(etavector(k)))-(1/2)*log2(2*pi/etavector(k))...
    -(1/2)*log2(exp(1))+(1/(2*beta))*log2(2*pi)+(1/(2*beta))*log2(exp(1));
end %for

n=50;  %50
T=100;   %100
nu=0.1;
Fenergy2=zeros(L,1);
Fenergy3=zeros(L,1);
Fenergy4=zeros(L,1);
for rep=1:L
beta=betavector(rep);
m=ceil(n/beta);
Phi_0=(1/sqrt(m))*randn(m,n); %initial
y_0=ones(m,1);
%Initialize
x_0=[Phi_0(:)' y_0']';  %inital state
t=1;
xcur2=x_0; 
xcur3=x_0; 
xcur4=x_0; 
d=length(x_0);
xnext2=zeros(d,1);
xnext3=zeros(d,1);
xnext4=zeros(d,1);
xstore2=zeros(d,T);
xstore3=zeros(d,T);
xstore4=zeros(d,T);
xstore2(:,1)=x_0;
xstore3(:,1)=x_0;
xstore4(:,1)=x_0;
 while t<=T
 x2=xcur2;
 x3=xcur3;
 x4=xcur4;
 [outpre12,out12]=P(x2,m,n,sigmanot,0.1);
 [outpre13,out13]=P(x3,m,n,sigmanot,0.5);
 [outpre14,out14]=P(x4,m,n,sigmanot,0.8);
 for i=1:d
   xnext2(i,1) = normrnd(x2(i,1),1);
   xnext3(i,1) = normrnd(x3(i,1),1);
   xnext4(i,1) = normrnd(x4(i,1),1);
 end
 [outpre2,out2]=P(xnext2,m,n,sigmanot,0.1);
 [outpre3,out3]=P(xnext3,m,n,sigmanot,0.5);
 [outpre4,out4]=P(xnext4,m,n,sigmanot,0.8);
 %xprime=mvnrnd(xprev,eye(d),1)'; 
 % Estimate acceptance probability 
 %P(x,m,n,alpha,delta)
 A2=min(1,out2/out12);
 A3=min(1,out3/out13);
 A4=min(1,out2/out14);
 % Accept and Reject
 U=rand;
  if U<=A2
    xcur2=xnext2;
  else
    xcur2=x2;
  end %if
  if U<=A3
    xcur3=xnext3;
  else
    xcur3=x3;
  end %if
  if U<=A4
    xcur4=xnext4;
  else
    xcur4=x4;
  end %if
 t=t+1;
 xstore2(:,t)=xcur2;
 xstore3(:,t)=xcur3;
 xstore4(:,t)=xcur4;
 end %while
% Monte-Carlo part, and estimating posterior distribution vector
 Posterdis2=zeros(T,1);
 Posterdis3=zeros(T,1);
 Posterdis4=zeros(T,1);
 for t=1:T
 [outpre,~]=P(xstore2(:,t),m,n,sigmanot,0.1);
 Posterdis2(t,1)= outpre; % P_{Y|Phi}
 [outpre,~]=P(xstore3(:,t),m,n,sigmanot,0.1);
 Posterdis3(t,1)= outpre; % P_{Y|Phi}
 [outpre,~]=P(xstore4(:,t),m,n,sigmanot,0.1);
 Posterdis4(t,1)= outpre; % P_{Y|Phi}
 end
Fenergy2(rep,1)=-(1/n)*mean(log2(Posterdis2(ceil(T/2):T)));
Fenergy3(rep,1)=-(1/n)*mean(log2(Posterdis3(ceil(T/2):T)));
Fenergy4(rep,1)=-(1/n)*mean(log2(Posterdis4(ceil(T/2):T)));
end %for
figure(1)
plot(betavector,F,'b-.',betavector,Fenergy2,'-r',betavector,Fenergy3,'--g',betavector,Fenergy4,'-.c','linewidth',2);
grid on
legend('Replica Method','MCMC (\nu=0.1)','MCMC (\nu=0.5)','MCMC (\nu=0.8)')
xlabel('\beta','fontsize',13)
ylabel('bits/symbol')
axis([0 5 0 8])


function [outpre,out]=P(x,m,n,sigmanot,nu)
% Divide x into [Phi, y] again
y=x(m*n+1:m*n+m,1);
Phi=reshape(x(1:m*n,1),[m,n]);
maxiter=100;
t=1;
F=zeros(maxiter,1);
while t<=maxiter %MC to estimate q_{Y|Phi}
% Generate a Markov Chain u^n
v=zeros(n,1);
v(1,1)=sqrt(sigmanot^2/(1-nu^2))*randn;
for i=2:n
     v(i,1)=nu*v(i-1,1)+randn*sigmanot;
end
%vecS=1+binornd(n,0.5);
%v=sqrt(diag(vecS))*v;
% Estimate q_{Y|Phi}
L1=((1/(2*pi))^(m/2))*exp(-(norm(y-Phi*v))^2/2); % P_{Y|Phi,X}
F(t)=L1; 
t=t+1;
end %while
outpre=mean(F); % P_{Y|Phi}
%out=((m/(2*pi))^(m*n/2))*exp(-(m/2)*(norm(Phi,'fro'))^2)*outpre; % P_{Phi Y}
out=exp((m/2)*(n*log(m/(2*pi))-(norm(Phi,'fro'))^2))*outpre;   % P_{Phi Y}
end
%function out = mylog2(in) %This function set xlog(x) to 0 as x to 0.
 % out = log2(in);
 % out(~in) = 0;
%end






