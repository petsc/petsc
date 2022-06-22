!
      program main
#include <petsc/finclude/petscksp.h>
      use petscksp
      implicit none

!
!  This example is the Fortran version of ex6.c.  The program reads a PETSc matrix
!  and vector from a file and solves a linear system.  Input arguments are:
!        -f <input_file> : file to load.  For example see $PETSC_DIR/share/petsc/datafiles/matrices
!

      PetscErrorCode  ierr
      PetscInt its,m,n,mlocal,nlocal
      PetscBool  flg
      PetscScalar      none
      PetscReal        norm
      Vec              x,b,u
      Mat              A
      character*(128)  f
      PetscViewer      fd
      MatInfo          info(MAT_INFO_SIZE)
      KSP              ksp

      none = -1.0
      PetscCallA(PetscInitialize(ierr))

! Read in matrix and RHS
      PetscCallA(PetscOptionsGetString(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-f',f,flg,ierr))
      PetscCallA(PetscViewerBinaryOpen(PETSC_COMM_WORLD,f,FILE_MODE_READ,fd,ierr))

      PetscCallA(MatCreate(PETSC_COMM_WORLD,A,ierr))
      PetscCallA(MatSetType(A, MATSEQAIJ,ierr))
      PetscCallA(MatLoad(A,fd,ierr))

! Get information about matrix
      PetscCallA(MatGetSize(A,m,n,ierr))
      PetscCallA(MatGetLocalSize(A,mlocal,nlocal,ierr))
      PetscCallA(MatGetInfo(A,MAT_GLOBAL_SUM,info,ierr))
      write(*,100) m,                                                   &
     &  n,                                                              &
     &  mlocal,nlocal,                                                  &
     &  info(MAT_INFO_BLOCK_SIZE),info(MAT_INFO_NZ_ALLOCATED),          &
     &  info(MAT_INFO_NZ_USED),info(MAT_INFO_NZ_UNNEEDED),              &
     &  info(MAT_INFO_MEMORY),info(MAT_INFO_ASSEMBLIES),                &
     &  info(MAT_INFO_MALLOCS)

 100  format(4(i4,1x),7(1pe9.2,1x))
      PetscCallA(VecCreate(PETSC_COMM_WORLD,b,ierr))
      PetscCallA(VecLoad(b,fd,ierr))
      PetscCallA(PetscViewerDestroy(fd,ierr))

! Set up solution
      PetscCallA(VecDuplicate(b,x,ierr))
      PetscCallA(VecDuplicate(b,u,ierr))

! Solve system
      PetscCallA(KSPCreate(PETSC_COMM_WORLD,ksp,ierr))
      PetscCallA(KSPSetOperators(ksp,A,A,ierr))
      PetscCallA(KSPSetFromOptions(ksp,ierr))
      PetscCallA(KSPSolve(ksp,b,x,ierr))

! Show result
      PetscCallA(MatMult(A,x,u,ierr))
      PetscCallA(VecAXPY(u,none,b,ierr))
      PetscCallA(VecNorm(u,NORM_2,norm,ierr))
      PetscCallA(KSPGetIterationNumber(ksp,its,ierr))
      write(6,101) norm,its
 101  format('Residual norm ',1pe9.2,' iterations ',i5)

! Cleanup
      PetscCallA(KSPDestroy(ksp,ierr))
      PetscCallA(VecDestroy(b,ierr))
      PetscCallA(VecDestroy(x,ierr))
      PetscCallA(VecDestroy(u,ierr))
      PetscCallA(MatDestroy(A,ierr))

      PetscCallA(PetscFinalize(ierr))
      end

!/*TEST
!
!    test:
!      args: -f ${DATAFILESPATH}/matrices/arco1 -options_left no
!      requires: datafilespath double  !complex !defined(PETSC_USE_64BIT_INDICES)
!
!TEST*/
