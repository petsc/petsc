C
C  Simple PETSc Program written in Fortran 77, (shudder)
C
       integer ierr

       PetscInitialize(ierr)

       if (ierr .ne. 0) then 
         print*, "Unable to begin PETSc program"
       endif

       print*, "Greetings"

       PetscFinalize()
       return
       end
