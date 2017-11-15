function[] = PetscCHKERRQ(err)
if (err)
   error('Errors generated on calling Petsc library function');
end