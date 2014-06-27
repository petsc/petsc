function err = PetscOptionsClearValue(option)
%
%  Removes an option from the database
%
err = calllib('libpetsc', 'PetscOptionsClearValue', option);PetscCHKERRQ(err);


