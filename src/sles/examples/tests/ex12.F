
C      "$Id: ex12.F,v 1.13 1996/03/06 03:30:21 bsmith Exp bsmith $";

#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"
#include "include/FINCLUDE/mat.h"
#include "include/FINCLUDE/pc.h"
#include "include/FINCLUDE/sles.h"
#include "include/FINCLUDE/viewer.h"
C
C  This example is the Fortran version of ex6.c.  The program reads a PETSc matrix
C  and vector from a file and solves a linear system.  Input arguments are:
C        -f <input_file> : file to load.  For a 5X5 example of the 5-pt. stencil
C                          use the file petsc/src/mat/examples/matbinary.ex
C

      integer          ierr, its, flg
      Scalar           time, norm,none
      Vec              x, b, u
      Mat              A
      SLES             sles
      character*(128)  f 
      Viewer           fd


      none = -1.0
      call PetscInitialize(PETSC_NULL_CHAR,ierr)

C Read in matrix and RHS
      call OptionsGetString(PETSC_NULL_CHAR,'-f',f,flg,ierr)
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

C Solve system
      call SLESCreate(MPI_COMM_WORLD,sles,ierr)
      call SLESSetOperators(sles,A,A,DIFFERENT_NONZERO_PATTERN,
     &                      ierr)
      call SLESSetFromOptions(sles,ierr)
      time = PetscGetTime()
      call SLESSolve(sles,b,x,its,ierr)
      time = PetscGetTime()-time

C Show result
      call MatMult(A,x,u,ierr)
      call VecAXPY(none,b,u,ierr)
      call VecNorm(u,NORM_2,norm,ierr)
      print*, 'Number of iterations = ',its
      print*, 'Residual norm = ',norm
      print*, 'Time for solve = ',time

C   /*   Cleanup   */
      call SLESDestroy(sles,ierr)
      call VecDestroy(b,ierr)
      call VecDestroy(x,ierr)
      call VecDestroy(u,ierr)
      call MatDestroy(A,ierr)

      call PetscFinalize(ierr)
      stop
      end

