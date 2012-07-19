function ex10(args)
%%
%
%  Mimics src/ksp/ksp/examples/tutorials/ex10.c loads a matrix from a binary file
%  and solves a linear system with it.
%
%   Set the Matlab path and initialize PETSc
path(path,'../../')
if (nargin == 0); args = {};end
PetscInitialize([args {'-ksp_monitor_true_residual','-ksp_converged_reason','-ksp_view'}]);
%%
%  Create a matrix and put some values in it
filename = PetscOptionsGetString('-f0');
filename 
viewer = PetscViewer(filename,Petsc.FILE_MODE_READ);
mat = PetscMat();
mat.SetFromOptions()
mat.Load(viewer);
viewer.Destroy();
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

%%
%  Write x into a binary file
%out_filename = %/path/out_filename. The default path might be ~MATLAB/
%viewer = PetscViewer(out_filename,Petsc.FILE_MODE_WRITE);
%x.View(viewer);
%viewer.Destroy();
%%
%   Free PETSc objects and shutdown PETSc
%
x.Destroy();
b.Destroy();
mat.Destroy();
ksp.Destroy();
PetscFinalize();
