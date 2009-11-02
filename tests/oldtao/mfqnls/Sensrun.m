global m nfev Fvals Xhist Fhist Deltahist delta % Global vars used in nls_f
h=1e-4;

n=9;
m=90;
nfev=0;
Fvals = zeros(2*n,1);
Xhist = zeros(2*n,n);
Fhist = zeros(2*n,m);
Deltahist = zeros(2*n,1);
delta=h;



x0=[0.419564390648983
   0.474248310575738
   0.484822039089875
   0.535912247739752
   0.406958773816197
   0.164281096207269
   0.478292446229684
   0.151415978598778     
   0.557700763396130]';



for i=1:n
   X =  x0;
   X(i) = X(i) + h;
   nls_f(X);
end
for i=1:n
   X =  x0;
   X(i) = X(i) - h;
   nls_f(X);
end
save('Hsens','Fhist','Xhist')
