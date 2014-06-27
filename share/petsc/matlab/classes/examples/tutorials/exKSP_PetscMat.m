%%
%
%  Solves a linear system where the user manages the mesh--solver interactions
%     User creates a MATLAB matrix and converts it to PETSc
%
%   Set the Matlab path and initialize PETSc
path(path,'../../')
PetscInitialize({'-ksp_monitor'});
%%
%   Open a viewer to display PETSc objects
viewer = PetscViewer();
viewer.SetType('ascii');
%%
%   Create a vector and put values in it
b = PetscVec();
b.SetType('seq');
b.SetSizes(10,10);
b.SetValues(1:10);
b.SetValues([1,2],[11.5,12.5],Petsc.ADD_VALUES);
b.AssemblyBegin();
b.AssemblyEnd();
b.View(viewer);
x = b.Duplicate();
%%
%  Create a MATLAB matrix and put some values in it
spmat = 10*speye(10,10);
spmat
mat = PetscMat(spmat);
mat.View(viewer);
mat(:,:) = speye(10,10);
mat.View(viewer);
%%
%   Create the linear solver, tell it the matrix to use and solve the system
ksp = PetscKSP();
ksp.SetType('gmres');
ksp.SetOperators(mat,mat,PetscMat.SAME_NONZERO_PATTERN);
ksp.SetFromOptions();
ksp.Solve(b,x);
x.View(viewer);
ksp.View(viewer);
%%
%   Free PETSc objects and Shutdown PETSc
%
x.Destroy();
b.Destroy();
mat.Destroy();
ksp.Destroy();
viewer.Destroy();
PetscFinalize();
