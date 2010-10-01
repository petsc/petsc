function err = PetscFinalize()
%
%
err = calllib('libpetsc', 'PetscFinalize');
unloadlibrary('libpetsc');

