!
!   This program tests MatGetDiagonal()
!
      program main
      implicit none

#include "include/finclude/petsc.h"
#include "include/finclude/petscis.h"
#include "include/finclude/petscvec.h"
#include "include/finclude/petscmat.h"
#include "include/finclude/petscviewer.h"

      integer ierr,i
      Vec     v
      Mat     m
      double precision value

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

      call MatCreate(PETSC_COMM_SELF,12,12,12,12,m,ierr)
      call MatSetFromOptions(m,ierr)

      value = 3.0
      i     = 4
      call MatSetValues(m,1,i,1,i,value,INSERT_VALUES,ierr)
      call MatAssemblyBegin(m,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(m,MAT_FINAL_ASSEMBLY,ierr)

      call VecCreateSeq(PETSC_COMM_SELF,12,v,ierr)
      call MatGetDiagonal(m,v,ierr)
      call VecView(v,PETSC_VIEWER_STDOUT_WORLD,ierr)

      call MatDestroy(m,ierr)
      call VecDestroy(v,ierr)
      call PetscFinalize(ierr)
      end


