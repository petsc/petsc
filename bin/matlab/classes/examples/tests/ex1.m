

% Test low level interface 

path(path,'../../')

PetscInitialize('-info');
PETSC_COMM_SELF = 1;

[err,vec] = calllib('libpetsc', 'VecCreate', PETSC_COMM_SELF,0)
err = calllib('libpetsc', 'VecSetType', vec,'seq')
err = calllib('libpetsc', 'VecSetSizes', vec,10,10)
  err = calllib('libpetsc', 'VecSetValues', vec,2,[1,2],[11.5,12.5],0)
err = calllib('libpetsc', 'VecAssemblyBegin', vec)
err = calllib('libpetsc', 'VecAssemblyEnd', vec)

[err,viewer] = calllib('libpetsc', 'PetscViewerCreate', PETSC_COMM_SELF,0)
err = calllib('libpetsc', 'PetscViewerSetType', viewer,'ascii')
err = calllib('libpetsc', 'VecView', vec,viewer)


