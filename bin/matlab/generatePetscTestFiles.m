% This MATLAB script generates test matrices for PETSc
% using PETSc-MATLAB IO functions and the function 
% http://www.mathworks.com/matlabcentral/fileexchange/27279-laplacian-in-1d-2d-or-3d

clear all; 
nx=3; ny=4; n=nx*ny;
% Create a nx-times-ny 2D negative Laplacian with h=1 and Dirichlet BC 
[~,~,A]=laplacian([nx,ny],{'DD' 'DD'});
x=ones(n,1); b=A*x; 

% this is the default
PetscBinaryWrite('spd-real-int32-float64',A,b,x, 'indices','int32','precision','float64');
[At,bt,xt] = PetscBinaryRead('spd-real-int32-float64', 'indices','int32','precision','float64');
max(max(abs(At-A)))

% int64 only appears in the header 
PetscBinaryWrite('spd-real-int64-float64',A, b,x,'indices','int64','precision','float64'); 
[At,bt,xt] = PetscBinaryRead('spd-real-int64-float64','indices','int64','precision','float64');
 max(max(abs(At-A)))

% MATLAB does not yet support single-presision sparce matrices
% but all the entries in A, b, and x are actually integers 
PetscBinaryWrite('spd-real-int32-float32',A,b,x, 'indices','int32','precision','float32');
[At,bt,xt] = PetscBinaryRead('spd-real-int32-float32', 'indices','int32','precision','float32');
max(max(abs(At-A)))

% int64 only appears in the header 
PetscBinaryWrite('spd-real-int64-float32',A,b,x,'indices','int64','precision','float32'); 
[At,bt,xt] = PetscBinaryRead('spd-real-int64-float32','indices','int64','precision','float32'); 
max(max(abs(At-A)))
 
% Now, we swap the (1,1) and (1,2) entries in A to make it nonsymmetric
% the solution x does not change and can be reused 
tmp=A(1,1); A(1,1)=A(1,2); A(1,2)=tmp; clear tmp;
PetscBinaryWrite('nonsymmetric-real-int32-float64',A,b,x,'indices','int32','precision','float64');
[At,bt,xt] = PetscBinaryRead('nonsymmetric-real-int32-float64','indices','int32','precision','float64');
max(max(abs(At-A)))
 
% Finally, we make A, b, and x complex, still integers
A(1,1)=A(1,1)+1i; A(2,1)=A(2,1)-1i;
x=1i.*ones(n,1); b=A*x; 
 
PetscBinaryWrite('nonhermitian-complex-int32-float64',A, b,x,'indices','int32','precision','float64');
[At,bt,xt] = PetscBinaryRead('nonhermitian-complex-int32-float64','indices','int32','precision','float64','complex',true);
max(max(abs(At-A)))

% Make A Hermitian, without changing b and x
tmp=A(1,1); A(1,1)=A(1,2); A(1,2)=tmp; clear tmp; 

% this is the complex default
PetscBinaryWrite('hpd-complex-int32-float64',A, b,x, 'indices','int32','precision','float64');
[At,bt,xt] = PetscBinaryRead('hpd-complex-int32-float64', 'indices','int32','precision','float64','complex',true);
max(max(abs(At-A)))

% int64 only appears in the header 
PetscBinaryWrite('hpd-complex-int64-float64',A,b,x, 'indices','int64','precision','float64'); 
[At,bt,xt] = PetscBinaryRead('hpd-complex-int64-float64', 'indices','int64','precision','float64','complex',true); 
max(max(abs(At-A)))
 
% MATLAB does not yet support single-presision sparce matrices
% but all the entries in A, b, and x are actually complex integers 
PetscBinaryWrite('hpd-complex-int32-float32',A,b,x, 'indices','int32','precision','float32');
[At,bt,xt] = PetscBinaryRead('hpd-complex-int32-float32','indices','int32','precision','float32','complex',true);
max(max(abs(At-A)))

% int64 only appears in the header 
PetscBinaryWrite('hpd-complex-int64-float32',A,b,x, 'indices','int64','precision','float32'); 
[At,bt,xt] = PetscBinaryRead('hpd-complex-int64-float32','indices','int64','precision','float32','complex',true); 
max(max(abs(At-A)))

