
C    "$Id: ex20.F,v 1.4 1996/04/12 23:41:07 curfman Exp balay $";
C
C  This example demonstrates repeated linear solves as well as the use of
C  different preconditioner and linear system matrices.  This example also
C  illustrates how to save PETSc objects in common blocks.
C
#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"
#include "include/FINCLUDE/mat.h"
#include "include/FINCLUDE/sles.h"
#include "include/FINCLUDE/pc.h"
#include "include/FINCLUDE/ksp.h"

      Vec     x, u, b
      Mat     A
      SLES    sles
      integer i, j, II, JJ, ierr, m, n
      integer Istart, Iend, flg, nsteps
      Scalar  v

      call PetscInitialize(PETSC_NULL_CHAR,ierr)
      m      = 3
      n      = 3
      nsteps = 2
      call OptionsGetInt(PETSC_NULL_CHAR,'-m',m,flg,ierr)
      call OptionsGetInt(PETSC_NULL_CHAR,'-n',n,flg,ierr)
      call OptionsGetInt(PETSC_NULL_CHAR,'-nsteps',nsteps,flg,ierr)

C Create and assemble initial matrix
      call MatCreate(MPI_COMM_WORLD,m*n,m*n,A,ierr)
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

C Create SLES context and set initial operators
      call SLESCreate(MPI_COMM_WORLD,sles,ierr)

C Set options
      call SLESSetFromOptions(sles,ierr)

C Solve several linear systems in succession
      do 100 i=1,nsteps
         call solve1(sles,A,x,b,u,i,nsteps,ierr)
 100  continue

C Free data structures
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
      subroutine solve1(sles,A,x,b,u,count,nsteps,ierr)

#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"
#include "include/FINCLUDE/mat.h"
#include "include/FINCLUDE/sles.h"
#include "include/FINCLUDE/pc.h"
#include "include/FINCLUDE/ksp.h"

C
C   solve1 - This routine is used for repeated linear system solves.
C   We update the linear system matrix each time, but retain the same
C   preconditioning matrix for all linear solves.
C
C      A - linear system matrix
C      A2 - preconditioning matrix
C
      Scalar  v, val
      integer II, ierr, Istart, Iend, count, nsteps, its
      Mat     A
      SLES    sles
      KSP     ksp
      Vec     x, b, u

C Use common block to retain matrix between successive subroutine calls
      Mat              A2
      integer          rank, pflag
      common /my_data/ A2, pflag, rank

C First time thorough: Create new matrix to define the linear system
      if (count .eq. 1) then
        call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)
        pflag = 0
        call OptionsHasName(PETSC_NULL_CHAR,"-mat_view",pflag,ierr)
        if (pflag .ne. 0) then
          if (rank .eq. 0) write(6,*) 'previous matrix: preconditioning'
        endif
        call MatConvert(A,MATSAME,A2,ierr)
C All other times: Set previous solution as initial guess for next solve.
      else
        call SLESGetKSP(sles,ksp,ierr)
        call KSPSetInitialGuessNonzero(ksp,ierr)
      endif

C Alter the matrix A a bit
      call MatGetOwnershipRange(A,Istart,Iend,ierr)
      do 20, II=Istart,Iend-1
        v = 2.0
        call MatSetValues(A,1,II,1,II,v,ADD_VALUES,ierr)
 20   continue
      call MatAssemblyBegin(A,FINAL_ASSEMBLY,ierr)
      if (pflag .ne. 0) then
        if (rank .eq. 0) write(6,*) 'next matrix: defines linear system'
      endif
      call MatAssemblyEnd(A,FINAL_ASSEMBLY,ierr)

C Set the exact solution; compute the right-hand-side vector
      val = 1.0*count
      call VecSet(val,u,ierr)
      call MatMult(A,u,b,ierr)

C Set operators, keeping the identical preconditioner matrix for
C all linear solves.  This approach is often effective when the
C linear systems do not change very much between successive steps.
      call SLESSetOperators(sles,A,A2,SAME_PRECONDITIONER,ierr)

C Solve linear system
      call SLESSolve(sles,b,x,its,ierr)

C Destroy the preconditioner matrix on the last time through
      if (count .eq. nsteps) call MatDestroy(A2,ierr)

      end

