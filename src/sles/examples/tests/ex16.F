!
!      "$Id: ex16.F,v 1.21 1998/12/05 21:08:39 balay Exp bsmith $";
!
       implicit none

#include "include/finclude/petsc.h"
#include "include/finclude/vec.h"
#include "include/finclude/mat.h"
#include "include/finclude/pc.h"
#include "include/finclude/sles.h"
#include "include/finclude/viewer.h"
#include "include/finclude/is.h"
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
      Scalar           norm,none,five
      Vec              x,b,u
      Mat              A
      SLES             sles1,sles2
      character*(128)  f 
      Viewer           fd
      IS               isrow
      PLogDouble       time1,time2

      none = -1.0
      five = 5.0
      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

! Read in matrix and RHS
      call OptionsGetString(PETSC_NULL_CHARACTER,'-f',f,flg,ierr)
      call ViewerBinaryOpen(PETSC_COMM_WORLD,f,BINARY_RDONLY,            &
     &     fd,ierr)
      if (ierr .ne. 0) then
        print*, 'Unable to open file ',f
        SETERRA(1,0,' ')
      endif

      call MatLoad(fd,MATSEQAIJ,A,ierr)
      if (ierr .ne. 0) then
        print*, 'Unable to load matrix '
        SETERRA(1,0,' ')
      endif

      call VecLoad(fd,b,ierr)
      call ViewerDestroy(fd,ierr)

! Set up solution
      call VecDuplicate(b,x,ierr)
      call VecDuplicate(b,u,ierr)

! Solve system-1
      call SLESCreate(PETSC_COMM_WORLD,sles1,ierr)
      call SLESSetOptionsPrefix(sles1,'a',ierr)
      call SLESAppendOptionsPrefix(sles1,'_',ierr)
      call SLESSetOperators(sles1,A,A,DIFFERENT_NONZERO_PATTERN,        &  
     &                      ierr)
      call SLESSetFromOptions(sles1,ierr)
      call PetscGetTime(time1,ierr)
      call SLESSolve(sles1,b,x,its,ierr)
      call PetscGetTime(time2,ierr)

! Show result
      call MatMult(A,x,u,ierr)
      call VecAXPY(none,b,u,ierr)
      call VecNorm(u,NORM_2,norm,ierr)

!      print*, 'Time for solve = ',time2-time1

      write(6,100) norm,its
  100 format('Residual norm ',e10.4,' iterations ',i5)

! Create system 2 by striping off some rows of the matrix
      call ISCreateStride(PETSC_COMM_SELF,5,0,1,isrow,ierr)
      call MatZeroRows(A,isrow,five,ierr)

! Solve system-2
      call SLESCreate(PETSC_COMM_WORLD,sles2,ierr)
      call SLESSetOptionsPrefix(sles2,'b',ierr)
      call SLESAppendOptionsPrefix(sles2,'_',ierr)
      call SLESSetOperators(sles2,A,A,DIFFERENT_NONZERO_PATTERN,        &
     &                      ierr)
      call SLESSetFromOptions(sles2,ierr)
      call PetscGetTime(time1,ierr)
      call SLESSolve(sles2,b,x,its,ierr)
      call PetscGetTime(time2,ierr)

! Show result
      call MatMult(A,x,u,ierr)
      call VecAXPY(none,b,u,ierr)
      call VecNorm(u,NORM_2,norm,ierr)
!      print*, 'Time for solve = ',time2-time1
      write(6,100) norm,its

! Cleanup
      call SLESDestroy(sles1,ierr)
      call SLESDestroy(sles2,ierr)
      call VecDestroy(b,ierr)
      call VecDestroy(x,ierr)
      call VecDestroy(u,ierr)
      call MatDestroy(A,ierr)
      call ISDestroy(isrow,ierr)

      call PetscFinalize(ierr)
      end

