nx   = 90;
ny   = 90;
nsdx =  3;
nsdy =  3;

xmin = 0.0;
xmax = 1.0;
ymin = 0.0;
ymax = 1.0;

y = linspace (ymin,ymax,ny+1);
x = linspace (xmin,xmax,nx+1);

% exact solution
[X,Y] = meshgrid(x,y);
U = (X-1).^2;

nelx = nx/nsdx;
nely = ny/nsdy;

solution; % load solution vector X

u = [];
p = 1;
for I = 1 : (nsdx-1)
   row = [];
   for J = 1 : (nsdy-1)
      n = nelx*nely;
      row = [row,(reshape(X(p:p+n-1),nely,nelx))'];
      p = p+n;
   end
   for J = nsdy : nsdy
      n = nelx*(nely+1);
      row = [row,(reshape(X(p:p+n-1),nely+1,nelx))'];
      p = p+n;
   end
   u = [u;row];
end
for I = nsdx : nsdx
   row = [];
   for J = 1 : (nsdy-1)
      n = (nelx+1)*nely;
      row = [row,(reshape(X(p:p+n-1),nely,nelx+1))'];
      p = p+n;
   end
   for J = nsdy :nsdy
      n = (nelx+1)*(nely+1);
      row = [row,(reshape(X(p:p+n-1),nely+1,nelx+1))'];
      p = p+n;
   end
   u = [u;row];
end

u = u';

figure(1)
surf(x,y,u)
title('Computed solution')
xlabel('x')
ylabel('y')
zlabel('z')

figure(2)
surf(x,y,abs(U-u))
title('Error (in absolute value)')
xlabel('x')
ylabel('y')
zlabel('z')
