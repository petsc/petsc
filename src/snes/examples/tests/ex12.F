C    "$Id: ex12.F,v 1.33 1996/04/15 22:09:27 bsmith Exp bsmith $";
C
C  This example demonstrates basic use of the SNES Fortran interface.
C  
C  Note:  The program ex10.f is the same as this example, except that it
C         uses the Fortran .f suffix rather than the .F suffix.
C
C  In this example the application context is a Fortran integer array:
C      ctx(1) = da    - distributed array
C          2  = F     - global vector where the function is stored
C          3  = xl    - local work vector
C          4  = rank  - processor rank
C          5  = size  - number of processors
C          6  = N     - system size
C
C  Note: Any user-defined Fortran routines (such as FormJacobian)
C  MUST be declared as external.
C

#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"
#include "include/FINCLUDE/da.h"
#include "include/FINCLUDE/mat.h"
#include "include/FINCLUDE/snes.h"

      integer          ctx(8),ierr,N,start,end,nn,i,ii,its,flg
      SNES             snes
      Mat              J
      Vec              x,r,u,F
      Scalar           xp,FF,UU,h
      character*(10)   matrixname
      external         FormJacobian, FormFunction

      call PetscInitialize(PETSC_NULL_CHAR,ierr)
      N = 10
      call OptionsGetInt(PETSC_NULL_CHAR,'-n',N,flg,ierr)
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

      call MatGetType(J,PETSC_NULL,matrixname,ierr)

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

C Set various routines and options
      call SNESSetFunction(snes,r,FormFunction,ctx,ierr)
      call SNESSetJacobian(snes,J,J,FormJacobian,ctx,ierr)
      call SNESSetFromOptions(snes,ierr)

C Solve nonlinear system
      call FormInitialGuess(snes,x,ierr)
      call SNESSolve(snes,x,its,ierr)

C Write results if first processor
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
      integer          snes, x, f, ctx(*)
      integer          rank, size, i, s, n, ierr, ixx, iff, iF2
      Scalar           h, d, vf2(1), vxx(1), vff(1)
#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"
#include "include/FINCLUDE/da.h"
#include "include/FINCLUDE/mat.h"
#include "include/FINCLUDE/snes.h"


      rank  = ctx(4)
      size  = ctx(5)
      h     = 1.d0/(ctx(6) - 1.d0)
      call DAGlobalToLocalBegin(ctx(1),x,INSERT_VALUES,ctx(3),ierr)
      call DAGlobalToLocalEnd(ctx(1),x,INSERT_VALUES,ctx(3),ierr)

      call VecGetLocalSize(ctx(3),n,ierr)
      if (n .gt. 1000) then 
        print*, 'Local work array not big enough'
        call MPI_Abort(MPI_COMM_WORLD,0,ierr)
      endif

C
C This sets the index ixx so that vxx(ixx+1) is the first local
C element in the vector indicated by ctx(3).
C
      call VecGetArray(ctx(3),vxx,ixx,ierr)
      call VecGetArray(f,vff,iff,ierr)
      call VecGetArray(ctx(2),vF2,iF2,ierr)
C
C Macros to make setting/getting  values into vector clearer.
C The element xx(ib) is the ibth element in the vector indicated by ctx(3)
#define xx(ib)  vxx(ixx + (ib))
#define ff(ib)  vff(iff + (ib))
#define F2(ib)  vF2(iF2 + (ib))

      d = h*h

      if (rank .eq. 0) then 
        s = 0
        ff(1) = xx(1)
      else
        s = 1
      endif

      do 10 i=1,n-2
       ff(i-s+1) = d*(xx(i) - 
     *                 2.d0*xx(i+1) + 
     *                 xx(i+2)) 
     *             + xx(i+1)*xx(i+1) -
     *               F2(i-s+1)
 10   continue

      if (rank .eq. size-1) then
        ff(n-s) = xx(n) - 1.d0
      endif

      call VecRestoreArray(f,vff,iff,ierr)
      call VecRestoreArray(ctx(3),vxx,ixx,ierr)
      call VecRestoreArray(ctx(2),vF2,iF2,ierr)
      return
      end

C --------------------  Form initial approximation ----------------- 

      subroutine FormInitialGuess(snes,x,ierr)
#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"
#include "include/FINCLUDE/snes.h"
      integer          ierr
      Vec              x
      SNES             snes
      Scalar           five 

      five = 5.d-1
      call VecSet(five,x,ierr)
      return
      end

C --------------------  Evaluate Jacobian -------------------- 

      subroutine FormJacobian(snes,x,jac,B,flag,ctx,ierr)
#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"
#include "include/FINCLUDE/da.h"
#include "include/FINCLUDE/mat.h"
#include "include/FINCLUDE/snes.h"
      SNES             snes
      Vec              x
      Mat              jac,B
      integer          flag, ctx(*), ii, ixx, istart
      integer          iend, i,j, n, rank, size, end, start, ierr
      Scalar           d, A, h, vxx(1)

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
      call MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY,ierr)
      return
      end



