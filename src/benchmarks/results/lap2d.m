%
%  Performance results for PETSc on the 2d Laplacian 5 pt stencil using GMRES and no 
% preconditioning. Done to give an indication of scalability for matrix vector product
% for very sparse matrix and scalability of inner products and vector updates.
%
p = [ 2 4 8 16 32;
m = [ 95.54 51.87 25.74 13.63 6.88; 134.6 67.59 33.93 16.77 9.03]; 
s = [455.5 245.2 123.9 72.41 38.63; 432.3 220 111.4 56.24 30.16];
%
