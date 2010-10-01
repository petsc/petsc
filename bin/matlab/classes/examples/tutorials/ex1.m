
%  Test high level interface

path(path,'../../')

PetscInitialize(1);

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


viewer.Destroy();


