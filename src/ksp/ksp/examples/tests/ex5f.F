!
      program main
#include <petsc/finclude/petscksp.h>
      use petscksp
      implicit none
!
!      Solves a linear system matrix free
!

      Mat         A
      Vec         x,y
      PetscInt  m
      PetscErrorCode ierr
      KSP        ksp
      external    mymatmult
      PetscScalar one

      m = 10

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif
      one = 1.0
      call KSPCreate(PETSC_COMM_SELF,ksp,ierr)

      call MatCreateShell(PETSC_COMM_SELF,m,m,m,m,0,A,ierr)
      call MatShellSetOperation(A,MATOP_MULT,mymatmult,ierr)

      call VecCreateSeq(PETSC_COMM_SELF,m,x,ierr)
      call VecDuplicate(x,y,ierr)
      call VecSet(x,one,ierr)

      call KSPSetOperators(ksp,A,A,ierr)
      call KSPSetFromOptions(ksp,ierr)

      call KSPSolve(ksp,x,y,ierr)

      call MatDestroy(A,ierr)
      call KSPDestroy(ksp,ierr)
      call VecDestroy(x,ierr)
      call VecDestroy(y,ierr)

      call PetscFinalize(ierr)
      end


!  This is a bogus multiply that copies the vector. This corresponds to
!  an identity matrix A

      subroutine mymatmult(A,x,y,ierr)
      use petscksp
      implicit none

      Mat A
      Vec x,y
      PetscErrorCode ierr

      call VecCopy(x,y,ierr)

      return
      end

!/*TEST
!
!    test:
!      args:  -ksp_monitor_short
!
!TEST*/
