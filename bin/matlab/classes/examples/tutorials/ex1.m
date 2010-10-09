
%  Test high level interface

path(path,'../../')

PetscInitialize({'-malloc_dump','-snes_monitor','-snes_mf_operator','-ksp_monitor'});

viewer = PetscViewer();
viewer.SetType('ascii');

vec = Vec;
vec.SetType('seq');
vec.SetSizes(10,10);
vec.SetValues(1:10);
vec.SetValues([1,2],[11.5,12.5],PetscObject.ADD_VALUES);
vec.AssemblyBegin();
vec.AssemblyEnd();
values = vec.GetValues([2 4])

% You can access PETSc Vec elements with regular Matlab indexing
vec([5 6])

% You can directly access all elements of a Vec
vec(:)

% You can assign PETSc Vec elements with regular Matlab indexing
vec(9) = 99;

vec.View(viewer);
vec.Destroy();

% You can directly create a PETSc Vec with a Matlab array
vec = Vec([2 3.1 4.5]);
vec.View(viewer);

vec.Destroy();

is = ISCreateGeneral([1 2 5]);
is.View(viewer);

mat = Mat;
mat.SetType('seqaij');
mat.SetSizes(10,10,10,10);
for i=0:9
  mat.SetValues(i,i,10.0);
end
mat.AssemblyBegin(Mat.FINAL_ASSEMBLY);
mat.AssemblyEnd(Mat.FINAL_ASSEMBLY);
mat.View(viewer);

b = Vec;
b.SetType('seq');
b.SetSizes(10,10);
b.SetValues(1:10);
b.SetValues([1,2],[11.5,12.5],PetscObject.ADD_VALUES);
b.AssemblyBegin();
b.AssemblyEnd();
x = b.Duplicate();

b.Copy(x);

ksp = KSP;
ksp.SetType('gmres');
ksp.SetOperators(mat,mat,Mat.SAME_NONZERO_PATTERN);
ksp.Solve(b,x);
x.View(viewer);
ksp.View(viewer);
ksp.Destroy();

arg = [1 2 4];
snes = SNES;
snes.SetType('ls');
snes.SetFunction(b,'nlfunction',arg);
snes.SetJacobian(mat,mat,'nljacobian',arg);
snes.SetFromOptions();
snes.Solve(x);
x.View(viewer);
snes.View(viewer);
snes.Destroy();

mat.Destroy();
b.Destroy();
x.Destroy();

viewer.Destroy();

arg
