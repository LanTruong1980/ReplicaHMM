% This program is written for the manuscript "Linear Models with Hidden
% Markov Sources via Replica Method" which was submitted to ISIT 2021.

clear
close all
clc
lambda=0.3 ;
gamma=0.8;
etavector=0.01:0.01:0.99;
L=length(etavector);
denobeta1=zeros(1,L);
denobeta2=zeros(1,L);
betasample=zeros(1,L);
for k=1:L
eta=etavector(k);
syms z;
ftempA=@(z) ((((eta*z/(1+eta))./(1+((1-gamma*lambda)/(lambda*gamma))*sqrt(1+eta)*exp(-eta^2*z.^2/(2*(1+eta)))))).^2)...
.*((1-gamma*lambda)*sqrt(eta/(2*pi))*exp(-eta*z.^2/2)+ lambda*gamma*sqrt(eta/(2*pi*(1+eta)))*exp(-eta*z.^2/(2*(1+eta))));   
denobeta1(k)=lambda-integral(ftempA,-80,80);
end
for k=1:L
eta=etavector(k);
syms z;
ftempB=@(z) ((((eta*z/(1+eta))./(1+((1-lambda)*gamma/(1-(1-lambda)*gamma))*sqrt(1+eta)*exp(-eta^2*z.^2/(2*(1+eta)))))).^2)...
.*((1-lambda)*gamma*sqrt(eta/(2*pi))*exp(-eta*z.^2/2)+ (1-(1-lambda)*gamma)*sqrt(eta/(2*pi*(1+eta)))*exp(-eta*z.^2/(2*(1+eta))));
denobeta2(k)=lambda-integral(ftempB,-80,80);
end
MMSE=zeros(1,L);
for k=1:L
eta=etavector(k);
MMSE(k)=(1-lambda)*denobeta1(k)+lambda*denobeta2(k);
betasample(k)=(1/eta-1)/MMSE(k);    
end

n=1000; % number of observations
betavector=betasample;
MSE=zeros(1,length(betavector));
for j=1:length(betavector)
beta=betavector(j);
m=floor(n/beta);
% Run Monte-Carlo simulation
maxrepeat=500;
rep=1;
MSEarray=zeros(1,maxrepeat);
while rep<maxrepeat
U=zeros(n,1);
X=zeros(n,1);
A=normrnd(0,1,[m,n]);
%Generate a Bernoulli Markov chain U_n.
U(1,1)=binornd(1,lambda);
for i=1:n-1
 if U(i,1)==0
   temp=binornd(1,lambda*gamma);
 else
   temp=binornd(1,1-(1-lambda)*gamma);
 end %if
 U(i+1,1)=temp;
end %for
% Form a hidden states sequence X_n based on U_n 
for i=1:n
    if U(i,1)==1
      X(i,1)=normrnd(0,1);
    else 
      X(i,1)=0;
    end
end
W=normrnd(0,1,[m,1]);
Y=(1/sqrt(m))*A*X+W;
iterAMP=10; % Number of Iterations
hatX=TurboAMP(n,beta,(1/sqrt(m))*A,Y,iterAMP,lambda);
temp=0;
for i=1:n
temp=temp+(hatX(i)-X(i))^2/n;
end
MSEarray(rep)=temp;
rep=rep+1;
end %while
MSE(j)=mean(MSEarray);
end



figure(1)
plot(betasample, MSE,'m-', betasample,MMSE,'b--','linewidth',2);
grid on
lg=legend('MSE for the Turbo AMP','MMSE (Replica Method)');
lg.Location='best';
xlabel('\beta','fontsize',13)
ylabel('MSE')
axis([1 4 0.1 0.3])
%axis([1 4 0.2 0.5])


function  xhat=TurboAMP(n,beta,A,y,iterAMP,lambda)
c=5; % c>>1;
z=y;
nu=zeros(n,1);
psi=zeros(n,1);
Fderi=zeros(n,1);
rep=0;
while rep<iterAMP
theta=A'*z+nu;
for l=1:n
[Fout,Gout,Fndout]=FnGnFnd(theta(l),c,lambda);
nu(l)=Fout;
psi(l)=Gout;
Fderi(l)=Fndout;
end
c=1+beta*mean(psi);
z=y-A*nu+beta*z*mean(Fderi);
rep=rep+1;
end %while
xhat=nu;
end


function [Fout,Gout,Fndout]=FnGnFnd(theta,c,lambda)
%This function estimate (48),(49), and (50) in Philip paper
alphanc=1/(c+1);
betanc=((1-lambda)/lambda)*((c+1)/c);
zetanc=1/(c*(c+1));
Fout=alphanc*theta/(1+betanc*exp(-zetanc*abs(theta)^2));
Gout=betanc*exp(-zetanc*abs(theta)^2)*abs(Fout)^2+(c/theta)*Fout;
Fndout=(alphanc/(1+betanc*exp(-zetanc*abs(theta)^2)))*(1+zetanc*abs(theta)^2/(1+(betanc*exp(-zetanc*abs(theta)^2))^(-1)));
end