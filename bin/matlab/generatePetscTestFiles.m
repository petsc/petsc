% This MATLAB script generates test matrices for PETSc
% using PETSc-MATLAB IO functions and the function laplacian.m from 
% http://www.mathworks.com/matlabcentral/fileexchange/27279-laplacian-in-1d-2d-or-3d


clear all; 
nx=3; ny=4; n=nx*ny; % any sizes can be used
% Create a nx-times-ny 2D negative Laplacian with h=1 and Dirichlet BC 
[~,~,A]=laplacian([nx,ny],{'DD' 'DD'});
% Alternatevely, a nx-times-ny-times-nz 3D negative Laplacian with h=1 and Dirichlet BC 
%nz=2; n=nx*ny*nz; [~,~,A]=laplacian([nx,ny,nz],{'DD' 'DD' 'DD'});

x=ones(n,1); b=A*x; %  use VecSet(x,1.0) in PETSc for x

% this is the default
PetscBinaryWrite('spd-real-int32-float64',A,b, 'indices','int32','precision','float64');
[At,bt] = PetscBinaryRead('spd-real-int32-float64', 'indices','int32','precision','float64');
if max(max(max(abs(At-A))),max(abs(bt-b))) ~= 0
 error('PETSc:generatePetscTestFiles:IncompatibleIO',...
 '%s','Error in PetscBinaryWrite or/and PetscBinaryRead for spd-real-int32-float64'); 
end

% int64 only appears in the header 
PetscBinaryWrite('spd-real-int64-float64',A, b,'indices','int64','precision','float64'); 
[At,bt] = PetscBinaryRead('spd-real-int64-float64','indices','int64','precision','float64');
if max(max(max(abs(At-A))),max(abs(bt-b))) ~= 0
 error('PETSc:generatePetscTestFiles:IncompatibleIO',...
 '%s','Error in PetscBinaryWrite or/and PetscBinaryRead for spd-real-int64-float64'); 
end

% MATLAB does not yet support single-presision sparce matrices
% but all the entries in A, b, and x are actually integers 
PetscBinaryWrite('spd-real-int32-float32',A,b, 'indices','int32','precision','float32');
[At,bt] = PetscBinaryRead('spd-real-int32-float32', 'indices','int32','precision','float32');
if max(max(max(abs(At-A))),max(abs(bt-b))) ~= 0
 error('PETSc:generatePetscTestFiles:IncompatibleIO',...
 '%s','Error in PetscBinaryWrite or/and PetscBinaryRead for spd-real-int32-float32'); 
end

% int64 only appears in the header 
PetscBinaryWrite('spd-real-int64-float32',A,b,'indices','int64','precision','float32'); 
[At,bt] = PetscBinaryRead('spd-real-int64-float32','indices','int64','precision','float32'); 
if max(max(max(abs(At-A))),max(abs(bt-b))) ~= 0
 error('PETSc:generatePetscTestFiles:IncompatibleIO',...
 '%s','Error in PetscBinaryWrite or/and PetscBinaryRead for spd-real-int64-float32'); 
end
 
% Now, we swap the (1,1) and (1,2) entries in A to make it nonsymmetric
tmp=A(1,1); A(1,1)=A(1,2); A(1,2)=tmp; clear tmp;
% the solution x does not change and can be reused. Check:
if max(abs(b-A*x))  ~= 0
 error('PETSc:generatePetscTestFiles:WrongAns',...
 '%s','The nonsymmetric matrix A is wrong'); 
end 

% We need to again write all possible formats
PetscBinaryWrite('ns-real-int32-float64',A,b,'indices','int32','precision','float64');
[At,bt] = PetscBinaryRead('ns-real-int32-float64','indices','int32','precision','float64');
if max(max(max(abs(At-A))),max(abs(bt-b))) ~= 0
 error('PETSc:generatePetscTestFiles:IncompatibleIO',...
 '%s','Error in PetscBinaryWrite or/and PetscBinaryRead for ns-real-int32-float64'); 
end
PetscBinaryWrite('ns-real-int64-float64',A,b,'indices','int64','precision','float64');
[At,bt] = PetscBinaryRead('ns-real-int64-float64','indices','int64','precision','float64');
if max(max(max(abs(At-A))),max(abs(bt-b))) ~= 0
 error('PETSc:generatePetscTestFiles:IncompatibleIO',...
 '%s','Error in PetscBinaryWrite or/and PetscBinaryRead for ns-real-int64-float64'); 
end
PetscBinaryWrite('ns-real-int32-float32',A,b,'indices','int32','precision','float32');
[At,bt] = PetscBinaryRead('ns-real-int32-float32','indices','int32','precision','float32');
if max(max(max(abs(At-A))),max(abs(bt-b))) ~= 0
 error('PETSc:generatePetscTestFiles:IncompatibleIO',...
 '%s','Error in PetscBinaryWrite or/and PetscBinaryRead for ns-real-int32-float32'); 
end
PetscBinaryWrite('ns-real-int64-float32',A,b,'indices','int64','precision','float32');
[At,bt] = PetscBinaryRead('ns-real-int64-float32','indices','int64','precision','float32');
if max(max(max(abs(At-A))),max(abs(bt-b))) ~= 0
 error('PETSc:generatePetscTestFiles:IncompatibleIO',...
 '%s','Error in PetscBinaryWrite or/and PetscBinaryRead for ns-real-int64-float32'); 
end

% Finally, we make A, b, and x complex, still integers
A(1,1)=A(1,1)+1i; A(2,1)=A(2,1)-1i;
x=1i.*ones(n,1); b=A*x;  % use VecSet(x,PETSC_i) in PETSc for x

% We need to again write all possible formats
PetscBinaryWrite('nh-complex-int32-float64',A,b,'indices','int32','precision','float64');
[At,bt] = PetscBinaryRead('nh-complex-int32-float64','indices','int32','precision','float64','complex',true);
if max(max(max(abs(At-A))),max(abs(bt-b))) ~= 0
 error('PETSc:generatePetscTestFiles:IncompatibleIO',...
 '%s','Error in PetscBinaryWrite or/and PetscBinaryRead for nh-complex-int32-float64'); 
end
PetscBinaryWrite('nh-complex-int64-float64',A,b,'indices','int64','precision','float64');
[At,bt] = PetscBinaryRead('nh-complex-int64-float64','indices','int64','precision','float64','complex',true);
if max(max(max(abs(At-A))),max(abs(bt-b))) ~= 0
 error('PETSc:generatePetscTestFiles:IncompatibleIO',...
 '%s','Error in PetscBinaryWrite or/and PetscBinaryRead for nh-complex-int64-float64'); 
end
PetscBinaryWrite('nh-complex-int32-float32',A,b,'indices','int32','precision','float32');
[At,bt] = PetscBinaryRead('nh-complex-int32-float32','indices','int32','precision','float32','complex',true);
if max(max(max(abs(At-A))),max(abs(bt-b))) ~= 0
 error('PETSc:generatePetscTestFiles:IncompatibleIO',...
 '%s','Error in PetscBinaryWrite or/and PetscBinaryRead for nh-complex-int32-float32'); 
end
PetscBinaryWrite('nh-complex-int64-float32',A,b,'indices','int64','precision','float32');
[At,bt] = PetscBinaryRead('nh-complex-int64-float32','indices','int64','precision','float32','complex',true);
if max(max(max(abs(At-A))),max(abs(bt-b))) ~= 0
 error('PETSc:generatePetscTestFiles:IncompatibleIO',...
 '%s','Error in PetscBinaryWrite or/and PetscBinaryRead for nh-complex-int64-float32'); 
end

% Make A Hermitian, without changing b and x
tmp=A(1,1); A(1,1)=A(1,2); A(1,2)=tmp; clear tmp; 
% the solution x does not change and can be reused. Check:
if max(max(abs(b-A*x)),max(max(abs(A-A'))))  ~= 0 
 error('PETSc:generatePetscTestFiles:WrongAns',...
 '%s','The HPD matrix A is wrong'); 
end 

% We need to again write all possible formats
PetscBinaryWrite('hpd-complex-int32-float64',A,b,'indices','int32','precision','float64');
[At,bt] = PetscBinaryRead('hpd-complex-int32-float64','indices','int32','precision','float64','complex',true);
if max(max(max(abs(At-A))),max(abs(bt-b))) ~= 0
 error('PETSc:generatePetscTestFiles:IncompatibleIO',...
 '%s','Error in PetscBinaryWrite or/and PetscBinaryRead for hpd-complex-int32-float64'); 
end
PetscBinaryWrite('hpd-complex-int64-float64',A,b,'indices','int64','precision','float64');
[At,bt] = PetscBinaryRead('hpd-complex-int64-float64','indices','int64','precision','float64','complex',true);
if max(max(max(abs(At-A))),max(abs(bt-b))) ~= 0
 error('PETSc:generatePetscTestFiles:IncompatibleIO',...
 '%s','Error in PetscBinaryWrite or/and PetscBinaryRead for hpd-complex-int64-float64'); 
end
PetscBinaryWrite('hpd-complex-int32-float32',A,b,'indices','int32','precision','float32');
[At,bt] = PetscBinaryRead('hpd-complex-int32-float32','indices','int32','precision','float32','complex',true);
if max(max(max(abs(At-A))),max(abs(bt-b))) ~= 0
 error('PETSc:generatePetscTestFiles:IncompatibleIO',...
 '%s','Error in PetscBinaryWrite or/and PetscBinaryRead for hpd-complex-int32-float32'); 
end
PetscBinaryWrite('hpd-complex-int64-float32',A,b,'indices','int64','precision','float32');
[At,bt] = PetscBinaryRead('hpd-complex-int64-float32','indices','int64','precision','float32','complex',true);
if max(max(max(abs(At-A))),max(abs(bt-b))) ~= 0
 error('PETSc:generatePetscTestFiles:IncompatibleIO',...
 '%s','Error in PetscBinaryWrite or/and PetscBinaryRead for hpd-complex-int64-float32'); 
end

