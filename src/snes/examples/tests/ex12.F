
#include "include/finclude/petsc.h"
#include "include/finclude/vec.h"
#include "include/finclude/da.h"
#include "include/finclude/mat.h"
#include "include/finclude/snes.h"
C
C      ctx(1) = da
C          2  = F
C          3  = xl
C          4  = mytid
C          5  = numtid
C          6  = N

      integer          ctx(8),ierr
      integer          snes,J,x,r,u,F,ierr,N,start,end,nn,i,ii,its
      double precision xp,FF,UU,h
      external         FormJacobian,FormFunction,FormInitialGuess

      call PetscInitialize(ierr)
      N = 10
      call OptionsGetInt(' ','-n',N,ierr)
      h = 1.d0/(N-1.d0)
      ctx(6) = N

      call  MPI_Comm_rank(MPI_COMM_WORLD,ctx(4),ierr)
      call  MPI_Comm_size(MPI_COMM_WORLD,ctx(5),ierr)

C     /* Set up data structures */
      call DACreate1d(MPI_COMM_WORLD,DA_NONPERIODIC,N,1,1,ctx(1),ierr)

      call DAGetDistributedVector(ctx(1),x,ierr)
      call DAGetLocalVector(ctx(1),ctx(3),ierr)

      call PetscObjectSetName(x,'Approximate Solution',ierr)
      call VecDuplicate(x,r,ierr)
      call VecDuplicate(x,F,ierr)
      ctx(2) = F
      call VecDuplicate(x,U,ierr)
      call  PetscObjectSetName(U,'Exact Solution',ierr)

      call MatCreateMPIAIJ(MPI_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,N,
     *                     N,3,0,0,0,J,ierr)

C     /* Store right-hand-side of PDE and exact solution */
      call VecGetOwnershipRange(x,start,end,ierr)
      xp = h*start
      nn = end - start
      ii = start
      do 10, i=0,nn-1
        FF = 6.d0*xp + (xp+1.d-12)**6.d0
        UU = xp*xp*xp
        call VecSetValues(F,1,ii,FF,INSERTVALUES,ierr)
        call VecSetValues(U,1,ii,UU,INSERTVALUES,ierr)
        xp = xp + h
        ii = ii + 1
 10   continue
      call VecAssemblyBegin(F,ierr)
      call VecAssemblyEnd(F,ierr)
      call VecAssemblyBegin(U,ierr)
      call VecAssemblyEnd(U,ierr)

C       /* Create nonlinear solver */  
      call SNESCreate(MPI_COMM_WORLD,SNES_NONLINEAR_EQUATIONS,snes,ierr)

C       /* Set various routines */
      call SNESSetSolution(snes,x,FormInitialGuess,0,ierr)
      call SNESSetFunction(snes,r,FormFunction,ctx,1,ierr)
      call SNESSetJacobian(snes,J,J,FormJacobian,ctx,ierr)

C      /* Set up nonlinear solver; then execute it */
      call SNESSetFromOptions(snes,ierr)
      call SNESSetUp(snes,ierr)
      call SNESSolve(snes,its,ierr)
      print*,'Number of Newton iterations = ', its

C  /* Free data structures */
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


C/* --------------------  Evaluate Function F(x) --------------------- */

      subroutine FormFunction(snes,x,f,ctx)
      integer snes,x,f,ctx(*)
      integer mytid, numtid,i,s,n,ierr
      double precision h,d,ff(1000),F2(1000),xx(1000)
#include "include/finclude/petsc.h"
#include "include/finclude/vec.h"
#include "include/finclude/da.h"
#include "include/finclude/mat.h"
#include "include/finclude/snes.h"

      mytid  = ctx(4)
      numtid = ctx(5)
      h = 1.d0/(ctx(6) - 1.d0)
      call DAGlobalToLocalBegin(ctx(1),x,INSERTVALUES,ctx(3),ierr)
      call DAGlobalToLocalEnd(ctx(1),x,INSERTVALUES,ctx(3),ierr)

      call VecGetLocalSize(ctx(3),n,ierr)
      if (n .gt. 1000) then 
        print*, 'Local work array not big enough'
        call MPI_Abort(MPI_COMM_WORLD,0,ierr)
      endif

      call VecGetArray(ctx(3),xx,ierr)
      call VecGetArray(f,ff,ierr)
      call VecGetArray(ctx(2),F2,ierr)

      d = h*h

      if (mytid .eq. 0) then 
        s = 0
        ff(1) = -xx(1)
      else
        s = 1
      endif

      do 10 i=1,n-2
       ff(i-s+1) = -d*(xx(i) - 2.d0*xx(i+1) + xx(i+2)) - xx(i+1)*xx(i+1)
     *             + F2(i-s+1)
 10   continue

      if (mytid .eq. numtid-1) then
        ff(n-s) = -xx(n) + 1.d0
      endif

      call VecRestoreArray(f,ff,ierr)
      return
      end

C  /* --------------------  Form initial approximation ----------------- */

      subroutine FormInitialGuess(snes,x,dummy,ierr)
      integer snes,x,dummy(*),ierr
      integer ierr
      double precision five 
#include "include/finclude/petsc.h"

      five = 5.d-1
      call VecSet(five,x,ierr)
      return
      end

C/* --------------------  Evaluate Jacobian F'(x) -------------------- */

      subroutine FormJacobian(snes,x,jac,B,flag,ctx,ierr)
#include "include/finclude/petsc.h"
#include "include/finclude/vec.h"
#include "include/finclude/da.h"
#include "include/finclude/mat.h"
#include "include/finclude/snes.h"
      integer          snes,x,jac(*),B,flag,ctx(*),N2,ii,istart,iend
      double precision xx(1000),d,A,h
      integer          i,j,ierr,n,mytid,numtid,end,start

      call VecGetSize(x,N2,ierr)
      h = 1.d0/(ctx(6) - 1.d0)
      d = h*h
      mytid = ctx(4)
      numtid = ctx(5)

      call VecGetArray(x,xx,ierr)
      call VecGetOwnershipRange(x,start,end,ierr)
      n = end - start

      if (mytid .eq. 0) then
        A = 1.0 
        call MatSetValues(jac(1),1,start,1,start,A,INSERTVALUES,ierr)
        istart = 1
      else 
        istart = 0
      endif
      if (mytid .eq. numtid-1) then
        i = N-1
        A = 1.0 
        call MatSetValues(jac(1),1,i,1,i,A,INSERTVALUES,ierr)
        iend = n-1
      else
        iend = n
      endif
      do 10 i=istart,iend
        ii = i + start
        j = start + i - 1 
        call MatSetValues(jac(1),1,ii,1,j,d,INSERTVALUES,ierr)
        j = start + i + 1 
        call MatSetValues(jac(1),1,ii,1,j,d,INSERTVALUES,ierr)
        A = -2.0*d + 2.0*xx(i+1)
        call MatSetValues(jac(1),1,ii,1,ii,A,INSERTVALUES,ierr)
 10   continue
      call MatAssemblyBegin(jac(1),FINAL_ASSEMBLY,ierr)
      callMatAssemblyEnd(jac(1),FINAL_ASSEMBLY,ierr)
      return
      end



