
function r = s1(x,y)
a1 = 30; a2 = 1;a3 = 5; a4 = 20;
eta = 0.04;
r = -2*eta*(a2+a4*y)/(a1+a2*x+a3*y+a4*x*y);

function r = s2(x,y)
a1 = 30; a2 = 1;a3 = 5; a4 = 20;
eta = 0.04;
r = 2*eta*(a3+a4*x)./ (a1+a2*x+a3*y+a4*x.*y);



