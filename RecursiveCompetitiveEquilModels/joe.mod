close all;
var k c n;
parameters delta alpha beta A; 

delta=0.2;
alpha=1.0/3.0;
beta=0.9; 
A = 1.0;

model;
c(+1) = beta*c*(alpha*(k^(alpha-1))*(n^(1-alpha))+(1-delta));
c + k = (A*k(-1)^alpha) * (n^(1-alpha)) + ((1-delta)*k(-1));
c = (1-n) * (1-alpha)  * A*k(-1)^alpha* n^(-alpha);
end;

initval;
c=0;
k=1.8;
n=1;
end;


endval;
c=5;
k=1;
n=0.5;
end;
steady;
check;

simul(periods=50);


tt=0:51;
figure
subplot(1,3,1);
plot(tt, oo_.endo_simul(1,:));
title('K');

subplot(1,3,2);
plot(tt, oo_.endo_simul(2,:));
title('C');

subplot(1,3,3);
plot(tt, oo_.endo_simul(3,:));
title('N');