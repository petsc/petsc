
C      "$Id: ex16.F,v 1.7 1996/02/24 17:25:45 curfman Exp bsmith $";

#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"
#include "include/FINCLUDE/mat.h"
#include "include/FINCLUDE/pc.h"
#include "include/FINCLUDE/sles.h"
#include "include/FINCLUDE/viewer.h"
#include "include/FINCLUDE/is.h"
C
C  This example is a modified Fortran version of ex6.c.  It tests the use of
C  options prefixes in PETSc. Two linear problems are solved in this program.
C  The first problem is read from a file. The second problem is constructed
C  from the first, by eliminating some of the entries of the linear matrix 'A'.

C  Each solve is distinguished by a unique prefix - 'a' for the first, 'b' 
C  for the second.  With the prefix the user can distinguish between the various
C  options (command line, from .petscrc file, etc.) for each of the solvers.
C  Input arguments are:
C     -f <input_file> : file to load.  For a 5X5 example of the 5-pt. stencil
C                       use the file petsc/src/mat/examples/mat.ex.binary

      integer          ierr, its, flg
      Scalar           time, norm,none, five
      Vec              x, b, u
      Mat              A
      SLES             sles1, sles2
      character*(128)  f 
      Viewer           fd
      IS               isrow

      none = -1.0
      five = 5.0
      call PetscInitialize(PETSC_NULL_CHAR,ierr)

C Read in matrix and RHS
      call OptionsGetString(PETSC_NULL_CHAR,'-f',f,flg,ierr)
      call ViewerFileOpenBinary(MPI_COMM_WORLD,f,BINARY_RDONLY,fd,ierr)
      if (ierr .ne. 0) then
        print*, 'Unable to open file ',f
        SETERRA(1,' ')
      endif

      call MatLoad(fd,MATSEQAIJ,A,ierr)
      if (ierr .ne. 0) then
        print*, 'Unable to load matrix '
        SETERRA(1,' ')
      endif

      call VecLoad(fd,b,ierr)
      call ViewerDestroy(fd,ierr)

C Set up solution
      call VecDuplicate(b,x,ierr)
      call VecDuplicate(b,u,ierr)

C Solve system-1
      call SLESCreate(MPI_COMM_WORLD,sles1,ierr)
      call SLESSetOptionsPrefix(sles1,'a', ierr)
      call SLESAppendOptionsPrefix(sles1,'_', ierr)
      call SLESSetOperators(sles1,A,A,DIFFERENT_NONZERO_PATTERN,
     &                      ierr)
      call SLESSetFromOptions(sles1,ierr)
      time = PetscGetTime()
      call SLESSolve(sles1,b,x,its,ierr)
      time = PetscGetTime()-time

C Show result
      call MatMult(A,x,u,ierr)
      call VecAXPY(none,b,u,ierr)
      call VecNorm(u,NORM_2,norm,ierr)

C      print*, 'Time for solve = ',time

      write(6,100) norm, its
  100 format('Residual norm ',e10.4,' iterations ',i5)

C Create system 2 by striping off some rows of the matrix
      call ISCreateStrideSeq(MPI_COMM_SELF,5,0,1,isrow, ierr)
      call MatZeroRows(A,isrow,five,ierr)

C Solve system-2
      call SLESCreate(MPI_COMM_WORLD,sles2,ierr)
      call SLESSetOptionsPrefix(sles2,'b', ierr)
      call SLESAppendOptionsPrefix(sles2,'_', ierr)
      call SLESSetOperators(sles2,A,A,DIFFERENT_NONZERO_PATTERN,
     &                      ierr)
      call SLESSetFromOptions(sles2,ierr)
      time = PetscGetTime()
      call SLESSolve(sles2,b,x,its,ierr)
      time = PetscGetTime()-time

C Show result
      call MatMult(A,x,u,ierr)
      call VecAXPY(none,b,u,ierr)
      call VecNorm(u,NORM_2,norm,ierr)
C      print*, 'Time for solve = ',time
      write(6,100) norm, its

C Cleanup
      call SLESDestroy(sles1,ierr)
      call SLESDestroy(sles2,ierr)
      call VecDestroy(b,ierr)
      call VecDestroy(x,ierr)
      call VecDestroy(u,ierr)
      call MatDestroy(A,ierr)
      call ISDestroy(isrow,ierr)

      call PetscFinalize(ierr)
      stop
      end

