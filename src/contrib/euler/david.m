 clf
 hold on
% z=1 logical plane
% Set a 2d view of x-y plane
view(0,90)

% axis([-1.5,1.5,0,1])
% axis('off')
%
duct
%
% View mesh only
% mesh(X,Y,Z)
%
% OR view contours for potential
% surf(X,Y,Z,potential)

% OR view mach contours 
 surf(X,Y,Z,mach)
% shading('interp')
% shading('flat')
title('Duct Problem: Potential');
%


