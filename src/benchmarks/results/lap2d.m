%
%  Performance results for PETSc on the 2d Laplacian 5 pt stencil using GMRES(30) and no 
% preconditioning. Run to give an indication of scalability for
%
%  * matrix vector product for very sparse matrix, and 
%  * scalability of inner products and vector updates.
%
% NOTE: one would NEVER use these particular solvers in practice.
%
% --------------------------------------------------------------------------
% For a 1000 by 1000 grid, with -ksp_max_it 500 -mat_aij_no_inode
%
p1000 = [ 2 4 8 16 32];
%
%  m and s contain the times for matrix vector multiply and SLES solve for 
%  IBM SP2, Cray T3E and Origin2000 in that order.
%
m1000 = [ 95.54 51.87 25.74 13.63 6.88; 134.6 67.59 33.93 16.77 9.03]; 
s1000 = [ 455.5 245.2 123.9 72.41 38.63; 432.3 220 111.4 56.24 30.16];
ms1000 = [ 95.54/95.54 95.54/51.87 95.54/25.74 95.54/13.63 95.54/6.88; 134.6/134.6 134.6/67.59 134.6/33.93 134.6/16.77 134.6/9.03; 1 2 4 8 16];
%
plotpt(p1000,ms1000,'*'); title('Speedup');
% --------------------------------------------------------------------------
% For a 100 by 100 grid, with -mat_aij_no_inode
%
p100 = [ 1 2 3 4 5 6 7 8];
%
%  m and s contain the times for matrix vector multiply and SLES solve for 
%  IBM SP2, Cray T3E and Origin2000 in that order.
%
m100 = [ 1.602 1.116 .793 .638 .5297 .4633 .4181 .5808; 2.233 1.434 1.064 .8116 .6748 .6095 .5338 .4873];
s100 = [ 9.425 6.036 4.32 3.334 3.634 3.656 3.657 4.248; 9.566 5.704 4.260 3.080 3.000 2.696 2.614 2.6061];
%
