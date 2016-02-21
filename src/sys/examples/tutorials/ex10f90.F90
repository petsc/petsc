      program ex10f90
#include "petsc/finclude/petscdef.h"
      use petsc
      implicit none

      PetscErrorCode                            :: ierr
      Character(len=256)                        :: filename
      PetscBool                                 :: flg

      Call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      call PetscOptionsGetString(PETSC_NULL_OBJECT,PETSC_NULL_CHARACTER,'-f',filename,flg,ierr)
      if (flg) then
         call PetscOptionsInsertFileYAML(PETSC_COMM_WORLD,filename,PETSC_TRUE,ierr);
      end if
      call PetscOptionsView(PETSC_NULL_OBJECT,PETSC_VIEWER_STDOUT_WORLD,ierr)
      Call PetscFinalize(ierr)
      end program ex10f90


