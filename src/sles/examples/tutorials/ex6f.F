
C    "$Id: ex7.F,v 1.9 1995/11/30 22:34:58 bsmith Exp $";

#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"
#include "include/FINCLUDE/mat.h"
#include "include/FINCLUDE/pc.h"
#include "include/FINCLUDE/sles.h"
C
C  This example demonstrates repeated linear solves as well as the use of
C  different preconditioner and linear system matrices.  This example also
C  illustrates how to save PETSc objects in common blocks.
C
      Vec              x, u, b
      Mat              A
      SLES             sles
      PetscRandom      rctx
      integer          i, j, II, JJ, ierr, m, n
      integer          Istart, Iend
      double precision v, zero, one

      zero = 0.0
      one  = 1.0
      call PetscInitialize(ierr)
      m = 3
      n = 3
      call OptionsGetInt(PETSC_NULL_CHAR,'-m',m,ierr)
      call OptionsGetInt(PETSC_NULL_CHAR,'-n',n,ierr)

C Create and assemble initial matrix
      call MatCreate(MPI_COMM_WORLD,m*n,m*n,,A,ierr)
      call MatGetOwnershipRange(A,Istart,Iend,ierr)
      do 10, II=Istart,Iend-1
        v = -1.0
        i = II/n
        j = II - i*n  
        if ( i.gt.0 ) then
          JJ = II - n
          call MatSetValues(A,1,II,1,JJ,v,ADD_VALUES,ierr)
        endif
        if ( i.lt.m-1 ) then
          JJ = II + n
          call MatSetValues(A,1,II,1,JJ,v,ADD_VALUES,ierr)
        endif
        if ( j.gt.0 ) then
          JJ = II - 1
          call MatSetValues(A,1,II,1,JJ,v,ADD_VALUES,ierr)
        endif
        if ( j.lt.n-1 ) then
          JJ = II + 1
          call MatSetValues(A,1,II,1,JJ,v,ADD_VALUES,ierr)
        endif
        v = 4.0
        call  MatSetValues(A,1,II,1,II,v,ADD_VALUES,ierr)
 10   continue
      call MatAssemblyBegin(A,FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(A,FINAL_ASSEMBLY,ierr)

C Create vectors for exact solution, approx solution, and RHS
      call VecCreateMPI(MPI_COMM_WORLD,PETSC_DECIDE,m*n,u,ierr)
      call VecDuplicate(u,b,ierr)
      call VecDuplicate(b,x,ierr)
      call PetscRandomCreate(MPI_COMM_WORLD,RANDOM_DEFAULT,rctx)
      call VecSetRandom(rctx,u,ierr)

C Create SLES context and set initial operators
      call SLESCreate(MPI_COMM_WORLD,sles,ierr)

C Set options
      call SLESSetFromOptions(sles,ierr)

C Solve several linear systems in succession
      do 100 i=1,nsteps
         call solve1(A,x,b,u,i,nsteps,ierr)
 100  continue

C Free data structures
      call PetscRandomDestroy(rctx,ierr)
      call VecDestroy(u,ierr)
      call VecDestroy(x,ierr)
      call VecDestroy(b,ierr)
      call MatDestroy(A,ierr)
      call SLESDestroy(sles,ierr)

      call PetscFinalize(ierr)
      stop
      end

C -----------------------------------------------------------------------
C
C   solve1 - This routine is used for repeated linear system solves.
C   We update the linear system matrix each time, but retain the same
C   preconditioning matrix for all linear solves.
C
C      A - linear system matrix
C      B - preconditioning matrix
C
#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"
#include "include/FINCLUDE/mat.h"
#include "include/FINCLUDE/pc.h"
#include "include/FINCLUDE/sles.h"

      subroutine solve1(A,x,b,u,count,nsteps,ierr)
      double precision v
      integer          II, ierr, Istart, Iend, count, nsteps, printflag
      Mat              A, A2
      SLES             sles

      common /my_data/ A2, printflag, rank

C First time thorough, create a new matrix to use as the preconditioner
      if (count .eq. 1) then
        printflag = 0
        call OptionsHasName(PETSC_NULL_CHAR,"-mat_view",printflag,ierr)
        if (rank .eq. 0) write(6,*) 'Preconditioning matrix'
        call MatConvert(A,MATSAME,A2,ierr)
      endif

C Alter the matrix A a bit
      call MatGetOwnershipRange(A,Istart,Iend,ierr)
      do 20, II=Istart,Iend-1
        v = 2.0
        call MatSetValues(A,1,II,1,II,v,ADD_VALUES,ierr)
 20   continue
      call MatAssemblyBegin(A,FINAL_ASSEMBLY,ierr)
      if (printflag .ne. 0) then
        if (rank .eq. 0) write(6,*) 'Stiffness matrix'
      endif
      call MatAssemblyEnd(A,FINAL_ASSEMBLY,ierr)

C Compute the right-hand-side vector
      call MatMult(A,u,b,ierr)

C Set operators, keeping the identical preconditioner matrix for
C all linear solves.  This approach is often effective when the
C linear systems do not change very much between successive steps.
C
      call SLESSetOperators(sles,A,A2,SAME_PRECONDITIONER,ierr)

C Solve linear system
      call SLESSolve(sles,b,x,its,ierr)

C Check the error
      none = -1.0
      call VecAXPY(none,u,x,ierr)
      call VecNorm(x,NORM_2,norm,ierr)
      if (rank .eq. 0) then
        if (norm .gt. 1.e-12) then
           write(6,100) norm, its
        else
           write(6,110) its
        endif
      endif
  100 format('Norm of error ',e10.4,' iterations ',i5)
  110 format('Norm of error < 1.e-12, iterations ',i5)

C Destroy the preconditioner matrix on the last time through
      if (count .eq. nsteps) call MatDestroy(B,ierr)

      end

