!
!   This program tests MatShellSetContext(), MatShellGetContext()
!
      program main
      implicit none

#include "include/finclude/petsc.h"
#include "include/finclude/petscis.h"
#include "include/finclude/petscvec.h"
#include "include/finclude/petscmat.h"
#include "include/finclude/petscviewer.h"

      PetscErrorCode ierr
      PetscInt i
      Vec     v,w
      Mat     m
      PetscInt ctx
      
      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

      ctx = 22
      call MatCreateShell(PETSC_COMM_SELF,12,12,12,12,ctx,m,ierr)
      ctx = 12
      call MatShellGetContext(m,ctx,ierr)
      print*, ctx

      call VecCreateSeq(PETSC_COMM_SELF,12,v,ierr)
      call VecCreateSeq(PETSC_COMM_SELF,12,w,ierr)
      call MatMult(m,v,w,ierr)

      call MatDestroy(m,ierr)
      call VecDestroy(v,ierr)
      call VecDestroy(w,ierr)
      call PetscFinalize(ierr)
      end


