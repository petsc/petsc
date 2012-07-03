!
!   This program tests MatGetVecs() for Shell Matrix
!
      subroutine mymatgetvecs(A,x,y,ierr)
      implicit none

#include <finclude/petscsys.h>
#include <finclude/petscis.h>
#include <finclude/petscvec.h>
#include <finclude/petscmat.h>
      PetscErrorCode ierr
      Mat A
      Vec x,y

      if (x .ne. -1) then
        call VecCreateSeq(PETSC_COMM_SELF,12,x,ierr)
      endif
      if (y .ne. -1) then
        call VecCreateSeq(PETSC_COMM_SELF,12,y,ierr)
      endif

      return
      end

      
      program main
      implicit none

#include <finclude/petscsys.h>
#include <finclude/petscis.h>
#include <finclude/petscvec.h>
#include <finclude/petscmat.h>
#include <finclude/petscviewer.h>

      PetscErrorCode ierr
      Vec     x,y
      Mat     m
      external  mymatgetvecs

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

      call MatCreateShell(PETSC_COMM_SELF,12,12,12,12,0,m,ierr)
      call MatAssemblyBegin(m,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(m,MAT_FINAL_ASSEMBLY,ierr)
      call MatShellSetOperation(m,MATOP_GET_VECS,mymatgetvecs,ierr)
      call MatGetVecs(m,x,y,ierr)
      call MatDestroy(m,ierr)
      call VecDestroy(x,ierr)
      call VecDestroy(y,ierr)
      call PetscFinalize(ierr)
      end


