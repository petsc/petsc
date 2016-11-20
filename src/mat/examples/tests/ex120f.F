!
!   This program tests MatCreateVecs() for Shell Matrix
!
      subroutine mymatgetvecs(A,x,y,ierr)
#include <petsc/finclude/petscmatdef.h>
      use petscmat
      implicit none

      PetscErrorCode ierr
      Mat A
      Vec x,y

      call VecCreateSeq(PETSC_COMM_SELF,12,x,ierr)
      call VecCreateSeq(PETSC_COMM_SELF,12,y,ierr)
      return
      end


      program main
#include <petsc/finclude/petscmatdef.h>
      use petscmat
      implicit none

      PetscErrorCode ierr
      Vec     x,y
      Mat     m
      external  mymatgetvecs

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif

      call MatCreateShell(PETSC_COMM_SELF,12,12,12,12,0,m,ierr)
      call MatAssemblyBegin(m,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(m,MAT_FINAL_ASSEMBLY,ierr)
      call MatShellSetOperation(m,MATOP_GET_VECS,mymatgetvecs,ierr)
      call MatCreateVecs(m,x,y,ierr)
      call MatDestroy(m,ierr)
      call VecDestroy(x,ierr)
      call VecDestroy(y,ierr)
      call PetscFinalize(ierr)
      end


