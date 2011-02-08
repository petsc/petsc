%%
%
%  Mimics src/ksp/ksp/examples/tutorials/ex10.c loads a matrix from a binary file
%  and solves a linear system with it.
%
%   Set the Matlab path and initialize PETSc
path(path,'../../')
PetscInitialize({'-ksp_monitor_true_residual','-ksp_converged_reason'});
%%
%  Create a matrix and put some values in it
filename = PetscOptionsGetString('-f0');
viewer = PetscViewer(filename,Petsc.FILE_MODE_READ);
mat = PetscMat();
mat.SetFromOptions()
mat.Load(viewer);
%%   Create a vector 
%
m = mat.GetSize();
b = PetscVec();
b.SetType('seq');
b.SetSizes(m,m);
b.Set(1.0);
x = b.Duplicate();

%%
%   Create the linear solver, tell it the matrix to use and solve the system
ksp = PetscKSP();
ksp.SetType('gmres');
ksp.SetOperators(mat,mat,PetscMat.SAME_NONZERO_PATTERN);
ksp.SetFromOptions();
ksp.Solve(b,x);
x.View;
ksp.View;
%%
%   Free PETSc objects and shutdown PETSc
%
x.Destroy();
b.Destroy();
mat.Destroy();
ksp.Destroy();
PetscFinalize();
