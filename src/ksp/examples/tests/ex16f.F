!
      program main
       implicit none

#include "include/finclude/petsc.h"
#include "include/finclude/petscvec.h"
#include "include/finclude/petscmat.h"
#include "include/finclude/petscpc.h"
#include "include/finclude/petscksp.h"
#include "include/finclude/petscviewer.h"
#include "include/finclude/petscis.h"
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

      integer          ierr,its,flg
      PetscScalar      norm,none,five
      Vec              x,b,u
      Mat              A
      KSP             ksp1,ksp2
      character*(128)  f 
      PetscViewer      fd
      IS               isrow
      PetscLogDouble   time1,time2
      KSP              ksp
      none = -1.0
      five = 5.0
      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

! Read in matrix and RHS
      call PetscOptionsGetString(PETSC_NULL_CHARACTER,'-f',f,flg,ierr)
      call PetscViewerBinaryOpen(PETSC_COMM_WORLD,f,PETSC_FILE_RDONLY,            &
     &     fd,ierr)
      if (ierr .ne. 0) then
        print*, 'Unable to open file ',f
        SETERRQ(1,' ',ierr)
      endif

      call MatLoad(fd,MATSEQAIJ,A,ierr)
      if (ierr .ne. 0) then
        print*, 'Unable to load matrix '
        SETERRQ(1,' ',ierr)
      endif

      call VecLoad(fd,PETSC_NULL_CHARACTER,b,ierr)
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
      call PetscGetTime(time1,ierr)
      call KSPSolve(ksp1,b,x,ierr)
      call PetscGetTime(time2,ierr)

! Show result
      call MatMult(A,x,u,ierr)
      call VecAXPY(none,b,u,ierr)
      call VecNorm(u,NORM_2,norm,ierr)
      call KSPGetIterationNumber(ksp,its,ierr)

!      print*, 'Time for solve = ',time2-time1

      write(6,100) norm,its
  100 format('Residual norm ',e10.4,' iterations ',i5)

! Create system 2 by striping off some rows of the matrix
      call ISCreateStride(PETSC_COMM_SELF,5,0,1,isrow,ierr)
      call MatZeroRows(A,isrow,five,ierr)

! Solve system-2
      call KSPCreate(PETSC_COMM_WORLD,ksp2,ierr)
      call KSPSetOptionsPrefix(ksp2,'b',ierr)
      call KSPAppendOptionsPrefix(ksp2,'_',ierr)
      call KSPSetOperators(ksp2,A,A,DIFFERENT_NONZERO_PATTERN,ierr)
      call KSPSetFromOptions(ksp2,ierr)
      call PetscGetTime(time1,ierr)
      call KSPSolve(ksp2,b,x,ierr)
      call PetscGetTime(time2,ierr)

! Show result
      call MatMult(A,x,u,ierr)
      call VecAXPY(none,b,u,ierr)
      call VecNorm(u,NORM_2,norm,ierr)
      call KSPGetIterationNumber(ksp,its,ierr)
!      print*, 'Time for solve = ',time2-time1
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

