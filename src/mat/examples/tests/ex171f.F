!
!   This program tests MatNullSpaceCreate()
!
      program main
      implicit none

#include <petsc/finclude/petscsys.h>
#include <petsc/finclude/petscis.h>
#include <petsc/finclude/petscvec.h>
#include <petsc/finclude/petscmat.h>
#include <petsc/finclude/petscviewer.h>

      PetscErrorCode ierr
      MatNullSpace nsp
      Vec     v(1)
      PetscInt nloc
      PetscScalar one
      PetscReal norm

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

      nloc = 12
      call VecCreateMPI(PETSC_COMM_WORLD,nloc,PETSC_DETERMINE,v(1),ierr)
      one = 1.0
      call VecSet(v(1),one,ierr)
      call VecNormalize(v(1),norm,ierr)
      call MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_FALSE,1,v(1),         &
     &                        nsp,ierr)
      call MatNullSpaceDestroy(nsp,ierr)
      call VecDestroy(v(1),ierr)
      call PetscFinalize(ierr)
      end


