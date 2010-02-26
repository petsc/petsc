!
      program main
       implicit none

#include "finclude/petscsys.h"
#include "finclude/petscvec.h"
#include "finclude/petscmat.h"
#include "finclude/petscpc.h"
#include "finclude/petscksp.h"
#include "finclude/petscviewer.h"
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
      one = 1.d0
      call KSPCreate(PETSC_COMM_SELF,ksp,ierr)

      call MatCreateShell(PETSC_COMM_SELF,m,m,m,m,PETSC_NULL_OBJECT,    &
     &     A,ierr)
      call MatShellSetOperation(A,MATOP_MULT,mymatmult,ierr)

      call VecCreateSeq(PETSC_COMM_SELF,m,x,ierr)
      call VecDuplicate(x,y,ierr)
      call VecSet(x,one,ierr)

      call KSPSetOperators(ksp,A,A,SAME_NONZERO_PATTERN,ierr)
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
     
      Mat A
      Vec x,y
      PetscErrorCode ierr
 
      call VecCopy(x,y,ierr)

      return 
      end
      
