
C      "$Id: ex12.F,v 1.1 1995/10/11 17:47:25 bsmith Exp bsmith $";

#include "include/finclude/petsc.h"
#include "include/finclude/vec.h"
#include "include/finclude/mat.h"
#include "include/finclude/pc.h"
#include "include/finclude/sles.h"
#include "include/finclude/viewer.h"
C
C  This example is the Fortran version of ex6.c
C

      integer          ierr, its
      double precision time, norm,none
      Vec              x, b, u
      Mat              A
      SLES             sles
      character*(128)  f 
      Viewer           fd

      none = -1.0
      call PetscInitialize(ierr)

C   /*   Read in matrix and RHS   */
      call OptionsGetString(' ','-f',f,ierr)
      call ViewerFileOpenBinary(MPI_COMM_WORLD,f,BINARY_RDONLY,fd,ierr)
      if (ierr .ne. 0) then
        print*, 'Unable to open file ',f
        SETERRA(1,' ')
      endif

      call MatLoad(fd,MATSEQAIJ,A,ierr)

      call VecLoad(fd,b,ierr)
      call ViewerDestroy(fd,ierr)

C   /*   Set up solution   */
      call VecDuplicate(b,x,ierr)
      call VecDuplicate(b,u,ierr)

C   /*   Solve system    */
      call SLESCreate(MPI_COMM_WORLD,sles,ierr)
      call SLESSetOperators(sles,A,A, ALLMAT_DIFFERENT_NONZERO_PATTERN,
     &                      ierr)
      call SLESSetFromOptions(sles,ierr)
      time = MPI_Wtime()
      call SLESSolve(sles,b,x,its,ierr)
      time = MPI_Wtime()-time

C   /*   Show result   */
      call MatMult(A,x,u,ierr)
      call VecAXPY(none,b,u,ierr)
      call VecNorm(u,norm,ierr)
      print*, 'Number of iterations = ',its
      print*, 'Residual norm = ',norm
      print*, 'Time for solve = ',time

C   /*   Cleanup   */
      call SLESDestroy(sles,ierr)
      call VecDestroy(b,ierr)
      call VecDestroy(u,ierr)
      call MatDestroy(A,ierr)

      call PetscFinalize(ierr)
      stop
      end

