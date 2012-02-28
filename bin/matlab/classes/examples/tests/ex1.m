

% Test low level interface 

path(path,'../../')

PetscInitialize('-info');

[err,vec] = calllib('libpetsc', 'VecCreate', 0,0)
err = calllib('libpetsc', 'VecSetType', vec,'seq')
err = calllib('libpetsc', 'VecSetSizes', vec,10,10)
  err = calllib('libpetsc', 'VecSetValues', vec,2,[1,2],[11.5,12.5],0)
err = calllib('libpetsc', 'VecAssemblyBegin', vec)
err = calllib('libpetsc', 'VecAssemblyEnd', vec)

[err,viewer] = calllib('libpetsc', 'PetscViewerCreate', 0,0)
err = calllib('libpetsc', 'PetscViewerSetType', viewer,'ascii')
err = calllib('libpetsc', 'VecView', vec,viewer)


