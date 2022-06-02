!
      program main
#include <petsc/finclude/petscksp.h>
      use petscksp
      implicit none

!
!  This example is a modified Fortran version of ex6.c.  It tests the use of
!  options prefixes in PETSc. Two linear problems are solved in this program.
!  The first problem is read from a file. The second problem is constructed
!  from the first, by eliminating some of the entries of the linear matrix 'A'.

!  Each solve is distinguished by a unique prefix - 'a' for the first, 'b'
!  for the second.  With the prefix the user can distinguish between the various
!  options (command line, from .petscrc file, etc.) for each of the solvers.
!  Input arguments are:
!     -f <input_file> : file to load.  For a 5X5 example of the 5-pt. stencil
!                       use the file petsc/src/mat/examples/mat.ex.binary

      PetscErrorCode                 ierr
      PetscInt                       its,ione,ifive,izero
      PetscBool                      flg
      PetscScalar                    none,five
      PetscReal                      norm
      Vec                            x,b,u
      Mat                            A
      KSP                            ksp1,ksp2
      character*(PETSC_MAX_PATH_LEN) f
      PetscViewer                    fd
      IS                             isrow
      none  = -1.0
      five  = 5.0
      ifive = 5
      ione  = 1
      izero = 0

      PetscCallA(PetscInitialize(ierr))

!     Read in matrix and RHS
      PetscCallA(PetscOptionsGetString(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-f',f,flg,ierr))
      PetscCallA(PetscViewerBinaryOpen(PETSC_COMM_WORLD,f,FILE_MODE_READ,fd,ierr))

      PetscCallA(MatCreate(PETSC_COMM_WORLD,A,ierr))
      PetscCallA(MatSetType(A, MATSEQAIJ,ierr))
      PetscCallA(MatLoad(A,fd,ierr))

      PetscCallA(VecCreate(PETSC_COMM_WORLD,b,ierr))
      PetscCallA(VecLoad(b,fd,ierr))
      PetscCallA(PetscViewerDestroy(fd,ierr))

! Set up solution
      PetscCallA(VecDuplicate(b,x,ierr))
      PetscCallA(VecDuplicate(b,u,ierr))

! Solve system-1
      PetscCallA(KSPCreate(PETSC_COMM_WORLD,ksp1,ierr))
      PetscCallA(KSPSetOptionsPrefix(ksp1,'a',ierr))
      PetscCallA(KSPAppendOptionsPrefix(ksp1,'_',ierr))
      PetscCallA(KSPSetOperators(ksp1,A,A,ierr))
      PetscCallA(KSPSetFromOptions(ksp1,ierr))
      PetscCallA(KSPSolve(ksp1,b,x,ierr))

! Show result
      PetscCallA(MatMult(A,x,u,ierr))
      PetscCallA(VecAXPY(u,none,b,ierr))
      PetscCallA(VecNorm(u,NORM_2,norm,ierr))
      PetscCallA(KSPGetIterationNumber(ksp1,its,ierr))

      write(6,100) norm,its
  100 format('Residual norm ',e11.4,' iterations ',i5)

! Create system 2 by striping off some rows of the matrix
      PetscCallA(ISCreateStride(PETSC_COMM_SELF,ifive,izero,ione,isrow,ierr))
      PetscCallA(MatZeroRowsIS(A,isrow,five,PETSC_NULL_VEC,PETSC_NULL_VEC,ierr))

! Solve system-2
      PetscCallA(KSPCreate(PETSC_COMM_WORLD,ksp2,ierr))
      PetscCallA(KSPSetOptionsPrefix(ksp2,'b',ierr))
      PetscCallA(KSPAppendOptionsPrefix(ksp2,'_',ierr))
      PetscCallA(KSPSetOperators(ksp2,A,A,ierr))
      PetscCallA(KSPSetFromOptions(ksp2,ierr))
      PetscCallA(KSPSolve(ksp2,b,x,ierr))

! Show result
      PetscCallA(MatMult(A,x,u,ierr))
      PetscCallA(VecAXPY(u,none,b,ierr))
      PetscCallA(VecNorm(u,NORM_2,norm,ierr))
      PetscCallA(KSPGetIterationNumber(ksp2,its,ierr))
      write(6,100) norm,its

! Cleanup
      PetscCallA(KSPDestroy(ksp1,ierr))
      PetscCallA(KSPDestroy(ksp2,ierr))
      PetscCallA(VecDestroy(b,ierr))
      PetscCallA(VecDestroy(x,ierr))
      PetscCallA(VecDestroy(u,ierr))
      PetscCallA(MatDestroy(A,ierr))
      PetscCallA(ISDestroy(isrow,ierr))

      PetscCallA(PetscFinalize(ierr))
      end

!/*TEST
!
!    test:
!      args: -f ${DATAFILESPATH}/matrices/arco1 -options_left no
!      requires: datafilespath double  !complex !defined(PETSC_USE_64BIT_INDICES)
!
!TEST*/
