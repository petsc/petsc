
C      "$Id: ex16.F,v 1.1 1996/01/16 19:30:06 balay Exp balay $";

#include "include/finclude/petsc.h"
#include "include/finclude/vec.h"
#include "include/finclude/mat.h"
#include "include/finclude/pc.h"
#include "include/finclude/sles.h"
#include "include/finclude/viewer.h"
#include "include/finclude/is.h"
C
C  This example is a modified Fortran version of ex6.c.  It tests the use of
C  options prefix in Petsc. Two linear problems are solved in this program. The first 
C  problem is read from a file. The second problem is constructed from the
C  first, by eliminating some of the entries of the linear matrix 'A'.

C  Each solve is distinguished by a unique prefix - 'a' for the first, 'b' 
C  for the second. Using the prefix, the user can distinguish between the 
C  options (command line, from .petscrc etc) for each of the solvers.
C  Input arguments are:
C        -f <input_file> : file to load.  For a 5X5 example of the 5-pt. stencil
C                          use the file petsc/src/mat/examples/mat.ex.binary
C

      integer          ierr, its, flg, five
      double precision time, norm,none
      Vec              x, b, u
      Mat              A
      SLES             sles1, sles2
      character*(128)  f 
      Viewer           fd
      IS               isrow

      none = -1.0
      five = 5.0
      call PetscInitialize(ierr)

C Read in matrix and RHS
      call OptionsGetString(PETSC_NULL,'-f',f,flg,ierr)
      call ViewerFileOpenBinary(MPI_COMM_WORLD,f,BINARY_RDONLY,fd,ierr)
      if (ierr .ne. 0) then
        print*, 'Unable to open file ',f
        SETERRA(1,' ')
      endif

      call MatLoad(fd,MATSEQAIJ,A,ierr)

      call VecLoad(fd,b,ierr)
      call ViewerDestroy(fd,ierr)

C Set up solution
      call VecDuplicate(b,x,ierr)
      call VecDuplicate(b,u,ierr)

C Solve system-1
      call SLESCreate(MPI_COMM_WORLD,sles1,ierr)
      call SLESSetOptionsPrefix(sles1,"a", ierr)
      call SLESAppendOptionsPrefix(sles1,"_", ierr)
      call SLESSetOperators(sles1,A,A, ALLMAT_DIFFERENT_NONZERO_PATTERN,
     &                      ierr)
      call SLESSetFromOptions(sles1,ierr)
      time = PetscGetTime()
      call SLESSolve(sles1,b,x,its,ierr)
      time = PetscGetTime()-time

C Show result
      call MatMult(A,x,u,ierr)
      call VecAXPY(none,b,u,ierr)
      call VecNorm(u,NORM_2,norm,ierr)
      print*, 'Number of iterations = ',its
      print*, 'Residual norm = ',norm
C      print*, 'Time for solve = ',time

C Create system 2 by striping off some rows of the matrix
/*      call ISCreateStrideSeq(MPI_COMM_SELF,5,0,1,isrow, ierr)
      call MatZeroRows(A, isrow, five, ierr)*/

C Solve system-2
      call SLESCreate(MPI_COMM_WORLD,sles2,ierr)
      call SLESSetOptionsPrefix(sles2,"b", ierr)
      call SLESAppendOptionsPrefix(sles2,"_", ierr)
      call SLESSetOperators(sles2,A,A, ALLMAT_DIFFERENT_NONZERO_PATTERN,
     &                      ierr)
      call SLESSetFromOptions(sles2,ierr)
      time = PetscGetTime()
      call SLESSolve(sles2,b,x,its,ierr)
      time = PetscGetTime()-time

C Show result
      call MatMult(A,x,u,ierr)
      call VecAXPY(none,b,u,ierr)
      call VecNorm(u,NORM_2,norm,ierr)
      print*, 'Number of iterations = ',its
      print*, 'Residual norm = ',norm
C      print*, 'Time for solve = ',time



C   /*   Cleanup   */
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

