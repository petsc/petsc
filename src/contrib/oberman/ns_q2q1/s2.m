function r = s2(x,y)
R = 10;
mu = .5*R + sqrt(.25*R*R + 4*pi*pi);	
r = -mu/(2*pi)*exp(-mu*x)*sin(2*pi*y);
