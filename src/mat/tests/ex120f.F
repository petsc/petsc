!
!   This program tests MatCreateVecs() for Shell Matrix
!
      subroutine mymatgetvecs(A,x,y,ierr)
#include <petsc/finclude/petscmat.h>
      use petscmat
      implicit none

      PetscErrorCode ierr
      Mat A
      Vec x,y
      PetscInt tw

      tw = 12
      call VecCreateSeq(PETSC_COMM_SELF,tw,x,ierr)
      call VecCreateSeq(PETSC_COMM_SELF,tw,y,ierr)
      return
      end

      program main
#include <petsc/finclude/petscmat.h>
      use petscmat
      implicit none

      PetscErrorCode ierr
      Vec     x,y
      Mat     m
      PetscInt tw
      external  mymatgetvecs

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif

      tw = 12
      call MatCreateShell(PETSC_COMM_SELF,tw,tw,tw,tw,0,m,ierr)
      call MatAssemblyBegin(m,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(m,MAT_FINAL_ASSEMBLY,ierr)
      call MatShellSetOperation(m,MATOP_CREATE_VECS,mymatgetvecs,ierr)
      call MatCreateVecs(m,x,y,ierr)
      call MatDestroy(m,ierr)
      call VecDestroy(x,ierr)
      call VecDestroy(y,ierr)
      call PetscFinalize(ierr)
      end

!/*TEST
!
!   test:
!      nsize: 2
!
!TEST*/
