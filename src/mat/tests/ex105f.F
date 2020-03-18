!
!   This program tests MatGetDiagonal()
!
      program main
#include <petsc/finclude/petscmat.h>
      use petscmat
      implicit none

      PetscErrorCode ierr
      PetscInt i,one,twelve,j
      Mat     m
      PetscScalar value

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif

      twelve = 12
      call MatCreate(PETSC_COMM_WORLD,m,ierr)
      call MatSetSizes(m,PETSC_DECIDE,PETSC_DECIDE,twelve,twelve,ierr)
      call MatSetFromOptions(m,ierr)
      call MatSetUp(m,ierr)

      value = 3.0
      i     = 4
      one   = 1
      call MatSetValuesMPIAIJ(m,one,i,one,i,value,ADD_VALUES,ierr)
      i = 5
      j = 7
      call MatSetValuesMPIAIJ(m,one,i,one,j,value,ADD_VALUES,ierr)
      i = 10
      j = 9
      call MatSetValuesMPIAIJ(m,one,i,one,j,value,ADD_VALUES,ierr)
      call MatAssemblyBegin(m,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(m,MAT_FINAL_ASSEMBLY,ierr)

      call MatDestroy(m,ierr)
      call PetscFinalize(ierr)
      end

!/*TEST
!
!   test:
!      nsize: 2
!
!TEST*/
