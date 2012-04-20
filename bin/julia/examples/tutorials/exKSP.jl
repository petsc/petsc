PetscInitialize(["-ksp_monitor","-malloc","-malloc_debug","-malloc_dump"]);
%%
%   Create a vector and put values in it
b = PetscVec();
PetscVecSetType(b,"seq");
PetscVecSetSizes(b,10,10);
PetscVecSetValues(b,float64(1.:10.));
PetscVecSetValues(b,[1,2],[11.5,12.5],PETSC_ADD_VALUES);
PetscVecAssemblyBegin(b);
PetscVecAssemblyEnd(b);
PetscView(b);

