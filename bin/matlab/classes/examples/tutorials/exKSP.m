%%
%
%  Solves a linear system where the user manages the mesh--solver interactions
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
b = Vec();
b.SetType('seq');
b.SetSizes(10,10);
b.SetValues(1:10);
b.SetValues([1,2],[11.5,12.5],PetscObject.ADD_VALUES);
b.AssemblyBegin();
b.AssemblyEnd();
b.View(viewer);
x = b.Duplicate();
%%
%  Create a matrix and put some values in it
mat = Mat();
mat.SetType('seqaij');
mat.SetSizes(10,10,10,10);
for i=0:9
  mat.SetValues(i,i,10.0);
end
mat.AssemblyBegin(Mat.FINAL_ASSEMBLY);
mat.AssemblyEnd(Mat.FINAL_ASSEMBLY);
mat.View(viewer);
%%
%   Create the linear solver, tell it the matrix to use and solve the system
ksp = KSP();
ksp.SetType('gmres');
ksp.SetOperators(mat,mat,Mat.SAME_NONZERO_PATTERN);
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
