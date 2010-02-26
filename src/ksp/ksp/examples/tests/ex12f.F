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
!  This example is the Fortran version of ex6.c.  The program reads a PETSc matrix
!  and vector from a file and solves a linear system.  Input arguments are:
!        -f <input_file> : file to load.  For a 5X5 example of the 5-pt. stencil
!                          use the file petsc/src/mat/examples/matbinary.ex
!

      PetscErrorCode  ierr
      PetscInt its,m,n,mlocal,nlocal
      PetscTruth flg
      PetscScalar      norm,none
      Vec              x,b,u
      Mat              A
      character*(128)  f 
      PetscViewer      fd
      MatInfo          info(MAT_INFO_SIZE)
      KSP              ksp

      none = -1.0
      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

! Read in matrix and RHS
      call PetscOptionsGetString(PETSC_NULL_CHARACTER,'-f',f,flg,ierr)
      print *,f
      call PetscViewerBinaryOpen(PETSC_COMM_WORLD,f,FILE_MODE_READ,     &
     &     fd,ierr)

      call MatLoad(fd,MATSEQAIJ,A,ierr)

! Get information about matrix
      call MatGetSize(A,m,n,ierr)
      call MatGetLocalSize(A,mlocal,nlocal,ierr)
      call MatGetInfo(A,MAT_GLOBAL_SUM,info,ierr)
      write(*,100) m,                                                   &
     &  n,                                                              &
     &  mlocal,nlocal,                                                  &
     &  info(MAT_INFO_BLOCK_SIZE),info(MAT_INFO_NZ_ALLOCATED),          &
     &  info(MAT_INFO_NZ_USED),info(MAT_INFO_NZ_UNNEEDED),              &
     &  info(MAT_INFO_MEMORY),info(MAT_INFO_ASSEMBLIES),                &
     &  info(MAT_INFO_MALLOCS)

 100  format(4(i4,1x),7(g7.1,1x))
      call VecLoad(fd,PETSC_NULL_CHARACTER,b,ierr)
      call PetscViewerDestroy(fd,ierr)

! Set up solution
      call VecDuplicate(b,x,ierr)
      call VecDuplicate(b,u,ierr)

! Solve system
      call KSPCreate(PETSC_COMM_WORLD,ksp,ierr)
      call KSPSetOperators(ksp,A,A,DIFFERENT_NONZERO_PATTERN,ierr)
      call KSPSetFromOptions(ksp,ierr)
      call KSPSolve(ksp,b,x,ierr)

! Show result
      call MatMult(A,x,u,ierr)
      call VecAXPY(u,none,b,ierr)
      call VecNorm(u,NORM_2,norm,ierr)
      call KSPGetIterationNumber(ksp,its,ierr)
      write(6,101) norm,its
 101  format('Residual norm ',e10.4,' iterations ',i5)

! Cleanup
      call KSPDestroy(ksp,ierr)
      call VecDestroy(b,ierr)
      call VecDestroy(x,ierr)
      call VecDestroy(u,ierr)
      call MatDestroy(A,ierr)

      call PetscFinalize(ierr)
      end

