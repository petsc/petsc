%%
% Solves a nonlinear system where the user manages the mesh--solver interactions
%
%   Set the Matlab path and initialize PETSc
path(path,'../../')
PetscInitialize({'-snes_monitor','-ksp_monitor'});
%%
%   Create work vector for nonlinear solver and location for solution
b = PetscVec();
b.SetType('seq');
b.SetSizes(10);
x = b.Duplicate();
%%
%  Create a matrix for the Jacobian for Newton method
mat = PetscMat();
mat.SetType('seqaij');
mat.SetSizes(10,10);
%%
%  Create the nonlinear solver 
snes = PetscSNES();
snes.SetType('vi');
%%
%  Provide a function 
snes.SetFunction(b,'nlfunction',0);
type nlfunction.m
%%
%  Provide a function that evaluates the Jacobian
snes.SetJacobian(mat,mat,'nljacobian',0);
type nljacobian.m
%%
%  Solve the nonlinear system
snes.SetFromOptions();
snes.Solve(x);
x.View;
snes.View;
%%
%   Free PETSc objects and shutdown PETSc
%
b.Destroy();
x.Destroy();
mat.Destroy();
snes.Destroy();
PetscFinalize();
