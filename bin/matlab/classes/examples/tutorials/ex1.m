
%  Test high level interface

path(path,'../../')

PetscInitialize({'-info','-malloc_dump'});

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
vec([5 6])

vec(9) = 99;

vec.View(viewer);
vec.Destroy();

vec = Vec([2 3.1 4.5]);
vec.View(viewer);
vec(:)
vec.Destroy();

is = ISCreateGeneral([1 2 5]);
is.View(viewer);

mat = Mat;
mat.SetType('seqaij');
mat.SetSizes(10,10,10,10);
mat.AssemblyBegin(Mat.MAT_FINAL_ASSEMBLY);
mat.AssemblyEnd(Mat.MAT_FINAL_ASSEMBLY);
mat.View(viewer);

viewer.Destroy();


