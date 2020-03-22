!
!   This program tests MatGetDiagonal()
!
      program main
#include <petsc/finclude/petscmat.h>
      use petscmat
      implicit none

      PetscErrorCode ierr
      PetscInt i,one,twelve
      Vec     v
      Mat     m
      PetscScalar value

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif

      twelve = 12
      call MatCreate(PETSC_COMM_SELF,m,ierr)
      call MatSetSizes(m,twelve,twelve,twelve,twelve,ierr)
      call MatSetFromOptions(m,ierr)
      call MatSetUp(m,ierr)

      value = 3.0
      i     = 4
      one   = 1
      call MatSetValues(m,one,i,one,i,value,INSERT_VALUES,ierr)
      call MatAssemblyBegin(m,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(m,MAT_FINAL_ASSEMBLY,ierr)

      call VecCreateSeq(PETSC_COMM_SELF,twelve,v,ierr)
      call MatGetDiagonal(m,v,ierr)
      call VecView(v,PETSC_VIEWER_STDOUT_WORLD,ierr)

      call MatDestroy(m,ierr)
      call VecDestroy(v,ierr)
      call PetscFinalize(ierr)
      end

!/*TEST
!
!   test:
!
!TEST*/
