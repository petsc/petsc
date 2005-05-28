!
      program main
       implicit none

#include "include/finclude/petsc.h"
#include "include/finclude/petscvec.h"
#include "include/finclude/petscmat.h"
#include "include/finclude/petscpc.h"
#include "include/finclude/petscksp.h"
#include "include/finclude/petscviewer.h"
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
      PetscInt m
      PetscErrorCode ierr
 
      m = 10

      call VecCopy(x,y,ierr)

      return 
      end
      
