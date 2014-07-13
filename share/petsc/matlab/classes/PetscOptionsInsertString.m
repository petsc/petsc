function err = PetscOptionsSetValue(options)
%
%  Adds one or more options to the database
%
err = calllib('libpetsc', 'PetscOptionsInsertString', options);PetscCHKERRQ(err);


