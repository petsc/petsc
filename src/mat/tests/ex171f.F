!
!   This program tests MatNullSpaceCreate()
!
      program main
#include <petsc/finclude/petscmat.h>
      use petscmat
      implicit none

      PetscErrorCode ierr
      MatNullSpace nsp
      Vec     v(1)
      PetscInt nloc,on
      PetscScalar one
      PetscReal norm

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif

      nloc = 12
      call VecCreateMPI(PETSC_COMM_WORLD,nloc,PETSC_DETERMINE,v(1),ierr)
      one = 1.0
      call VecSet(v(1),one,ierr)
      call VecNormalize(v(1),norm,ierr)
      on = 1
      call MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_FALSE,on,v(1),       &
     &                        nsp,ierr)
      call MatNullSpaceDestroy(nsp,ierr)
      call VecDestroy(v(1),ierr)
      call PetscFinalize(ierr)
      end

!/*TEST
!
!   test:
!
!TEST*/
