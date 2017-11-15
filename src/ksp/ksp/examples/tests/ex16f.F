!
      program main
       implicit none

#include <finclude/petscsys.h>
#include <finclude/petscvec.h>
#include <finclude/petscmat.h>
#include <finclude/petscpc.h>
#include <finclude/petscksp.h>
#include <finclude/petscviewer.h>
#include <finclude/petscis.h>
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

      PetscErrorCode  ierr
      PetscInt its,ione,ifive,izero
      PetscErrorCode flg
      PetscScalar      norm,none,five
      Vec              x,b,u
      Mat              A
      KSP             ksp1,ksp2
      character*(128)  f
      PetscViewer      fd
      IS               isrow
      none  = -1.0
      five  = 5.0
      ifive = 5
      ione  = 1
      izero = 0

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

! Read in matrix and RHS
      call PetscOptionsGetString(PETSC_NULL_CHARACTER,'-f',f,flg,ierr)
      call PetscViewerBinaryOpen(PETSC_COMM_WORLD,f,FILE_MODE_READ,            &
     &     fd,ierr)
      if (ierr .ne. 0) then
        print*, 'Unable to open file ',f
        SETERRQ(PETSC_COMM_WORLD,1,' ',ierr)
      endif

      call MatCreate(PETSC_COMM_WORLD,A,ierr)
      call MatSetType(A, MATSEQAIJ,ierr)
      call MatLoad(A,fd,ierr)
      if (ierr .ne. 0) then
        print*, 'Unable to load matrix '
        SETERRQ(PETSC_COMM_WORLD,1,' ',ierr)
      endif

      call VecCreate(PETSC_COMM_WORLD,b,ierr)
      call VecLoad(b,fd,ierr)
      call PetscViewerDestroy(fd,ierr)

! Set up solution
      call VecDuplicate(b,x,ierr)
      call VecDuplicate(b,u,ierr)

! Solve system-1
      call KSPCreate(PETSC_COMM_WORLD,ksp1,ierr)
      call KSPSetOptionsPrefix(ksp1,'a',ierr)
      call KSPAppendOptionsPrefix(ksp1,'_',ierr)
      call KSPSetOperators(ksp1,A,A,DIFFERENT_NONZERO_PATTERN,ierr)
      call KSPSetFromOptions(ksp1,ierr)
      call KSPSolve(ksp1,b,x,ierr)

! Show result
      call MatMult(A,x,u,ierr)
      call VecAXPY(u,none,b,ierr)
      call VecNorm(u,NORM_2,norm,ierr)
      call KSPGetIterationNumber(ksp1,its,ierr)


      write(6,100) norm,its
  100 format('Residual norm ',e10.4,' iterations ',i5)

! Create system 2 by striping off some rows of the matrix
      call ISCreateStride(PETSC_COMM_SELF,ifive,izero,ione,isrow,ierr)
      call MatZeroRowsIS(A,isrow,five,PETSC_NULL_OBJECT,                   &
     &                   PETSC_NULL_OBJECT,ierr)

! Solve system-2
      call KSPCreate(PETSC_COMM_WORLD,ksp2,ierr)
      call KSPSetOptionsPrefix(ksp2,'b',ierr)
      call KSPAppendOptionsPrefix(ksp2,'_',ierr)
      call KSPSetOperators(ksp2,A,A,DIFFERENT_NONZERO_PATTERN,ierr)
      call KSPSetFromOptions(ksp2,ierr)
      call KSPSolve(ksp2,b,x,ierr)

! Show result
      call MatMult(A,x,u,ierr)
      call VecAXPY(u,none,b,ierr)
      call VecNorm(u,NORM_2,norm,ierr)
      call KSPGetIterationNumber(ksp2,its,ierr)
      write(6,100) norm,its

! Cleanup
      call KSPDestroy(ksp1,ierr)
      call KSPDestroy(ksp2,ierr)
      call VecDestroy(b,ierr)
      call VecDestroy(x,ierr)
      call VecDestroy(u,ierr)
      call MatDestroy(A,ierr)
      call ISDestroy(isrow,ierr)

      call PetscFinalize(ierr)
      end

