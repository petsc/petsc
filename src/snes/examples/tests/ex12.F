C    "$Id: ex12.F,v 1.17 1995/12/06 00:25:54 bsmith Exp bsmith $";

#include "include/finclude/petsc.h"
#include "include/finclude/vec.h"
#include "include/finclude/da.h"
#include "include/finclude/mat.h"
#include "include/finclude/snes.h"
C
C  This example demonstrates basic use of the SNES Fortran interface.
C  In this example the application context is a Fortran integer array:
C
C      ctx(1) = da
C          2  = F
C          3  = xl
C          4  = rank
C          5  = size
C          6  = N
C
      integer          ctx(8),ierr,N,start,end,nn,i,ii,its
      SNES             snes
      Mat              J
      Vec              x,r,u,F
      double precision xp,FF,UU,h
      character*(10)   matrixname
      external         FormJacobian,FormFunction,FormInitialGuess

      call PetscInitialize(ierr)
      N = 10
      call OptionsGetInt(PETSC_NULL,'-n',N,ierr)
      h = 1.d0/(N-1.d0)
      ctx(6) = N

      call MPI_Comm_rank(MPI_COMM_WORLD,ctx(4),ierr)
      call MPI_Comm_size(MPI_COMM_WORLD,ctx(5),ierr)

C Set up data structures
      call DACreate1d(MPI_COMM_WORLD,DA_NONPERIODIC,N,1,1,ctx(1),ierr)

      call DAGetDistributedVector(ctx(1),x,ierr)
      call DAGetLocalVector(ctx(1),ctx(3),ierr)

      call PetscObjectSetName(x,'Approximate Solution',ierr)
      call VecDuplicate(x,r,ierr)
      call VecDuplicate(x,F,ierr)
      ctx(2) = F
      call VecDuplicate(x,U,ierr)
      call PetscObjectSetName(U,'Exact Solution',ierr)

      call MatCreateMPIAIJ(MPI_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,N,
     *                     N,3,PETSC_NULL,0,PETSC_NULL,J,ierr)

      call MatGetName(J,matrixname,ierr)
C     print*, 'Matrix name ',matrixname

C Store right-hand-side of PDE and exact solution
      call VecGetOwnershipRange(x,start,end,ierr)
      xp = h*start
      nn = end - start
      ii = start
      do 10, i=0,nn-1
        FF = 6.d0*xp + (xp+1.d-12)**6.d0
        UU = xp*xp*xp
        call VecSetValues(F,1,ii,FF,INSERT_VALUES,ierr)
        call VecSetValues(U,1,ii,UU,INSERT_VALUES,ierr)
        xp = xp + h
        ii = ii + 1
 10   continue
      call VecAssemblyBegin(F,ierr)
      call VecAssemblyEnd(F,ierr)
      call VecAssemblyBegin(U,ierr)
      call VecAssemblyEnd(U,ierr)

C Create nonlinear solver
      call SNESCreate(MPI_COMM_WORLD,SNES_NONLINEAR_EQUATIONS,snes,ierr)

C Set various routines
      call SNESSetSolution(snes,x,FormInitialGuess,0,ierr)
      call SNESSetFunction(snes,r,FormFunction,ctx,
     *                     NEGATIVE_FUNCTION_VALUE,ierr)
      call SNESSetJacobian(snes,J,J,FormJacobian,ctx,ierr)

C Set up nonlinear solver; then execute it
      call SNESSetFromOptions(snes,ierr)
      call SNESSetUp(snes,ierr)
      call SNESSolve(snes,its,ierr)
      if (ctx(4) .eq. 0) then
        write(6,100) its
      endif
  100 format('Number of Newton iterations = ',i5)

C Free data structures
      call VecDestroy(x,ierr)
      call VecDestroy(ctx(3),ierr)
      call VecDestroy(r,ierr)
      call VecDestroy(U,ierr)
      call VecDestroy(F,ierr)
      call MatDestroy(J,ierr)
      call SNESDestroy(snes,ierr)
      call DADestroy(ctx(1),ierr)
      call PetscFinalize(ierr)
      stop
      end


C --------------------  Evaluate Function F(x) --------------------- 

      subroutine FormFunction(snes,x,f,ctx)
      integer snes, x, f, ctx(*)
      integer rank, size, i, s, n, ierr, ixx, iff, iF2
      double precision h, d, vf2(1), vxx(1), vff(1)
#include "include/finclude/petsc.h"
#include "include/finclude/vec.h"
#include "include/finclude/da.h"
#include "include/finclude/mat.h"
#include "include/finclude/snes.h"

#define ff(ib)  vff(iff + (ib))
#define F2(ib)  vF2(iF2 + (ib))
#define xx(ib)  vxx(ixx + (ib))

      rank  = ctx(4)
      size = ctx(5)
      h = 1.d0/(ctx(6) - 1.d0)
      call DAGlobalToLocalBegin(ctx(1),x,INSERT_VALUES,ctx(3),ierr)
      call DAGlobalToLocalEnd(ctx(1),x,INSERT_VALUES,ctx(3),ierr)

      call VecGetLocalSize(ctx(3),n,ierr)
      if (n .gt. 1000) then 
        print*, 'Local work array not big enough'
        call MPI_Abort(MPI_COMM_WORLD,0,ierr)
      endif

      call VecGetArray(ctx(3),vxx,ixx,ierr)
      call VecGetArray(f,vff,iff,ierr)
      call VecGetArray(ctx(2),vF2,iF2,ierr)

      d = h*h

      if (rank .eq. 0) then 
        s = 0
        ff(1) = -xx(1)
      else
        s = 1
      endif

      do 10 i=1,n-2
       ff(i-s+1) = -d*(xx(i) - 
     *                 2.d0*xx(i+1) + 
     *                 xx(i+2)) 
     *             - xx(i+1)*xx(i+1) +
     *               F2(i-s+1)
 10   continue

      if (rank .eq. size-1) then
        ff(n-s) = -xx(n) + 1.d0
      endif

      call VecRestoreArray(f,vff,iff,ierr)
      call VecRestoreArray(ctx(3),vxx,ixx,ierr)
      call VecRestoreArray(ctx(2),vF2,iF2,ierr)
      return
      end

C --------------------  Form initial approximation ----------------- 

      subroutine FormInitialGuess(snes,x,dummy,ierr)
      integer snes,x,dummy(*),ierr
      double precision five 
#include "include/finclude/petsc.h"

      five = 5.d-1
      call VecSet(five,x,ierr)
      return
      end

C --------------------  Evaluate Jacobian -------------------- 

      subroutine FormJacobian(snes,x,jac,B,flag,ctx,ierr)
#include "include/finclude/petsc.h"
#include "include/finclude/vec.h"
#include "include/finclude/da.h"
#include "include/finclude/mat.h"
#include "include/finclude/snes.h"
      integer          snes, x, jac(*), B, flag, ctx(*), ii, ixx, istart
      integer          iend, i,j, n, rank, size, end, start, ierr
      double precision d, A, h, vxx(1)

#define xx(ib)  vxx(ixx + (ib))

      h = 1.d0/(ctx(6) - 1.d0)
      d = h*h
      rank = ctx(4)
      size = ctx(5)

      call VecGetArray(x,vxx,ixx,ierr)
      call VecGetOwnershipRange(x,start,end,ierr)
      n = end - start

      if (rank .eq. 0) then
        A = 1.0 
        call MatSetValues(jac,1,start,1,start,A,INSERT_VALUES,ierr)
        istart = 1
      else 
        istart = 0
      endif
      if (rank .eq. size-1) then
        i = ctx(6)-1
        A = 1.0 
        call MatSetValues(jac,1,i,1,i,A,INSERT_VALUES,ierr)
        iend = n-1
      else
        iend = n
      endif
      do 10 i=istart,iend-1
        ii = i + start
        j = start + i - 1 
        call MatSetValues(jac,1,ii,1,j,d,INSERT_VALUES,ierr)
        j = start + i + 1 
        call MatSetValues(jac,1,ii,1,j,d,INSERT_VALUES,ierr)
        A = -2.0*d + 2.0*xx(i+1)
        call MatSetValues(jac,1,ii,1,ii,A,INSERT_VALUES,ierr)
 10   continue
      call VecRestoreArray(x,vxx,ixx,ierr)
      call MatAssemblyBegin(jac,FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(jac,FINAL_ASSEMBLY,ierr)
C      call MatView(jac,STDOUT_VIEWER_WORLD)
      return
      end



