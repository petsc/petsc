!
!      "$Id: ex12.F,v 1.37 2001/08/07 03:03:57 balay Exp $";
!
      program main
       implicit none

#include "include/finclude/petsc.h"
#include "include/finclude/petscvec.h"
#include "include/finclude/petscmat.h"
#include "include/finclude/petscpc.h"
#include "include/finclude/petscsles.h"
#include "include/finclude/petscviewer.h"
!
!  This example is the Fortran version of ex6.c.  The program reads a PETSc matrix
!  and vector from a file and solves a linear system.  Input arguments are:
!        -f <input_file> : file to load.  For a 5X5 example of the 5-pt. stencil
!                          use the file petsc/src/mat/examples/matbinary.ex
!

      integer          ierr,its,flg
      PetscScalar      norm,none
      Vec              x,b,u
      Mat              A
      SLES             sles
      character*(128)  f 
      PetscViewer      fd
      MatInfo          info(MAT_INFO_SIZE)
      PetscLogDouble   time1,time2
      KSP              ksp

      none = -1.0
      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

! Read in matrix and RHS
      call PetscOptionsGetString(PETSC_NULL_CHARACTER,'-f',f,flg,ierr)
      call PetscViewerBinaryOpen(PETSC_COMM_WORLD,f,PETSC_FILE_RDONLY,
     &     fd,ierr)

      call MatLoad(fd,MATSEQAIJ,A,ierr)

! Get information about matrix
      call MatGetInfo(A,MAT_GLOBAL_SUM,info,ierr)
      write(6,100) info(MAT_INFO_ROWS_GLOBAL),                          &
     &  info(MAT_INFO_COLUMNS_GLOBAL),                                  &
     &  info(MAT_INFO_ROWS_LOCAL),info(MAT_INFO_COLUMNS_LOCAL),         &
     &  info(MAT_INFO_BLOCK_SIZE),info(MAT_INFO_NZ_ALLOCATED),          &
     &  info(MAT_INFO_NZ_USED),info(MAT_INFO_NZ_UNNEEDED),              &
     &  info(MAT_INFO_MEMORY),info(MAT_INFO_ASSEMBLIES),                &
     &  info(MAT_INFO_MALLOCS)

 100  format(11(g7.1,1x))
      call VecLoad(fd,b,ierr)
      call PetscViewerDestroy(fd,ierr)

! Set up solution
      call VecDuplicate(b,x,ierr)
      call VecDuplicate(b,u,ierr)

! Solve system
      call SLESCreate(PETSC_COMM_WORLD,sles,ierr)
      call SLESSetOperators(sles,A,A,DIFFERENT_NONZERO_PATTERN,         &
     &                      ierr)
      call SLESSetFromOptions(sles,ierr)
      call PetscGetTime(time1,ierr)
      call SLESSolve(sles,b,x,ierr)
      call PetscGetTime(time2,ierr)

! Show result
      call MatMult(A,x,u,ierr)
      call VecAXPY(none,b,u,ierr)
      call VecNorm(u,NORM_2,norm,ierr)
      call SLESGetKSP(sles,ksp,ierr)
      call KSPGetIterationNumber(ksp,its,ierr)
      print*, 'Number of iterations = ',its
      print*, 'Residual norm = ',norm
      print*, 'Time for solve = ',time2-time1

! Cleanup
      call SLESDestroy(sles,ierr)
      call VecDestroy(b,ierr)
      call VecDestroy(x,ierr)
      call VecDestroy(u,ierr)
      call MatDestroy(A,ierr)

      call PetscFinalize(ierr)
      end

