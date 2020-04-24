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
      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif

! Read in matrix and RHS
      call PetscOptionsGetString(PETSC_NULL_OPTIONS,                    &
     &        PETSC_NULL_CHARACTER,'-f',f,flg,ierr)
      call PetscViewerBinaryOpen(PETSC_COMM_WORLD,f,FILE_MODE_READ,     &
     &     fd,ierr)

      call MatCreate(PETSC_COMM_WORLD,A,ierr)
      call MatSetType(A, MATSEQAIJ,ierr)
      call MatLoad(A,fd,ierr)

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

 100  format(4(i4,1x),7(1pe9.2,1x))
      call VecCreate(PETSC_COMM_WORLD,b,ierr)
      call VecLoad(b,fd,ierr)
      call PetscViewerDestroy(fd,ierr)

! Set up solution
      call VecDuplicate(b,x,ierr)
      call VecDuplicate(b,u,ierr)

! Solve system
      call KSPCreate(PETSC_COMM_WORLD,ksp,ierr)
      call KSPSetOperators(ksp,A,A,ierr)
      call KSPSetFromOptions(ksp,ierr)
      call KSPSolve(ksp,b,x,ierr)

! Show result
      call MatMult(A,x,u,ierr)
      call VecAXPY(u,none,b,ierr)
      call VecNorm(u,NORM_2,norm,ierr)
      call KSPGetIterationNumber(ksp,its,ierr)
      write(6,101) norm,its
 101  format('Residual norm ',1pe9.2,' iterations ',i5)

! Cleanup
      call KSPDestroy(ksp,ierr)
      call VecDestroy(b,ierr)
      call VecDestroy(x,ierr)
      call VecDestroy(u,ierr)
      call MatDestroy(A,ierr)

      call PetscFinalize(ierr)
      end

!/*TEST
!
!    test:
!      args: -f ${DATAFILESPATH}/matrices/arco1 -options_left no
!      requires: datafilespath double  !complex !define(PETSC_USE_64BIT_INDICES)
!
!TEST*/
