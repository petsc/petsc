
C      "$Id: ex12.F,v 1.15 1996/08/22 22:43:52 curfman Exp curfman $";

       implicit none

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
      MatInfo          info(MAT_INFO_SIZE)

      none = -1.0
      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

C Read in matrix and RHS
      call OptionsGetString(PETSC_NULL_CHARACTER,'-f',f,flg,ierr)
      call ViewerFileOpenBinary(MPI_COMM_WORLD,f,BINARY_RDONLY,fd,ierr)
      if (ierr .ne. 0) then
        print*, 'Unable to open file ',f
        SETERRA(1,' ')
      endif

      call MatLoad(fd,MATSEQAIJ,A,ierr)

      call MatGetInfo(A,MAT_GLOBAL_SUM,info,ierr)
      write(6,100) info(1),info(2),info(3),info(4),info(5),info(6),
     &             info(7),info(8),info(9),info(10),info(11),
     &             info(12),info(13),info(14)
 100  format(14(g7.1,1x))
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

