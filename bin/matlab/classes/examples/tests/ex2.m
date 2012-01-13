
%  Test high level interface

path(path,'../../')

PetscInitialize({'-malloc_dump','-snes_monitor','-ksp_monitor'});

viewer = PetscViewer();
viewer.SetType('ascii');

PetscOptionsView(viewer);

vec = PetscVec();
vec.SetType('seq');
vec.SetSizes(10,10);
vec.SetValues(1:10);
vec.SetValues([1,2],[11.5,12.5],Petsc.ADD_VALUES);
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
vec = PetscVec([2 3.1 4.5]);
vec.View(viewer);
vec.Destroy();

% You can create an IS directly with a matlab array
is = PetscIS([1 2 5]);
% You can directly access all elements of an IS
indices = is(:);
is.View(viewer);
is.Destroy();

mat = PetscMat();
mat.SetType('seqaij');
mat.SetSizes(10,10,10,10);
for i=1:10
  mat.SetValues(i,i,2*i);
end

mat.AssemblyBegin(PetscMat.FINAL_ASSEMBLY);
mat.AssemblyEnd(PetscMat.FINAL_ASSEMBLY);
mat.View(viewer);

b = PetscVec();
b.SetType('seq');
b.SetSizes(10,10);
b.SetValues(1:10);
b.SetValues([1,2],[11.5,12.5],Petsc.ADD_VALUES);
b.AssemblyBegin();
b.AssemblyEnd();
x = b.Duplicate();

b.Copy(x);

ksp = PetscKSP();
ksp.SetType('gmres');
ksp.SetOperators(mat,mat,PetscMat.SAME_NONZERO_PATTERN);
ksp.Solve(b,x);
x.View(viewer);
ksp.View(viewer);
ksp.Destroy();

arg = [1 2 4];
snes = PetscSNES();
snes.SetType('ls');
snes.SetFunction(b,'ex2_nlfunction',arg);
snes.SetJacobian(mat,mat,'ex2_nljacobian',arg);
snes.SetFromOptions();
snes.Solve(x);
x.View(viewer);
snes.View(viewer);
snes.Destroy();

mat.Destroy();
b.Destroy();
x.Destroy();

da = PetscDMDACreate1d(PetscDM.BOUNDARY_NONE,10,1,1);
da.View(viewer);

ksp = PetscKSP();
ksp.SetDM(da);
ksp.Destroy();
da.Destroy();

viewer.Destroy();

ts = PetscTS();
ts.SetFromOptions();
ts.Destroy();

PetscFinalize();
