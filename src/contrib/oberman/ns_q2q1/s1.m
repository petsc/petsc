function r = s1(x,y)
R = 10;
mu = .5*R + sqrt(.25*R*R + 4*pi*pi);	
r = 1 - exp(-mu*x)*cos(2*pi*y);
  
