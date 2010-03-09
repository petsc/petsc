!
!
!  This example demonstrates basic use of the SNES Fortran interface.
!  
!  Note:  The program ex10.f is the same as this example, except that it
!         uses the Fortran .f suffix rather than the .F suffix.
!
!  In this example the application context is a Fortran integer array:
!      ctx(1) = da    - distributed array
!          2  = F     - global vector where the function is stored
!          3  = xl    - local work vector
!          4  = rank  - processor rank
!          5  = size  - number of processors
!          6  = N     - system size
!
!  Note: Any user-defined Fortran routines (such as FormJacobian)
!  MUST be declared as external.
!
!
! Macros to make setting/getting  values into vector clearer.
! The element xx(ib) is the ibth element in the vector indicated by ctx(3)
#define xx(ib)  vxx(ixx + (ib))
#define ff(ib)  vff(iff + (ib))
#define F2(ib)  vF2(iF2 + (ib))
      program main
      implicit none

#include "finclude/petscsys.h"
#include "finclude/petscvec.h"
#include "finclude/petscda.h"
#include "finclude/petscmat.h"
#include "finclude/petscsnes.h"

      PetscFortranAddr ctx(6)
      PetscMPIInt rank,size
      PetscErrorCode ierr
      PetscInt N,start,end,nn,i
      PetscInt ii,its,i1,i0,i3
      PetscTruth flg
      SNES             snes
      Mat              J
      Vec              x,r,u
      PetscScalar      xp,FF,UU,h
      character*(10)   matrixname
      external         FormJacobian,FormFunction

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      i1 = 1
      i0 = 0
      i3 = 3
      N  = 10
      call PetscOptionsGetInt(PETSC_NULL_CHARACTER,'-n',N,flg,ierr)
      h = 1.d0/(N-1.d0)
      ctx(6) = N

      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)
      call MPI_Comm_size(PETSC_COMM_WORLD,size,ierr)
      ctx(4) = rank
      ctx(5) = size

! Set up data structures
      call DACreate1d(PETSC_COMM_WORLD,DA_NONPERIODIC,N,i1,i1,            &
     &     PETSC_NULL_INTEGER,ctx(1),ierr)

      call DACreateGlobalVector(ctx(1),x,ierr)
      call DACreateLocalVector(ctx(1),ctx(3),ierr)

      call PetscObjectSetName(x,'Approximate Solution',ierr)
      call VecDuplicate(x,r,ierr)
      call VecDuplicate(x,ctx(2),ierr)
      call VecDuplicate(x,U,ierr)
      call PetscObjectSetName(U,'Exact Solution',ierr)

      call MatCreateMPIAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,N, &
     &     N,i3,PETSC_NULL_INTEGER,i0,PETSC_NULL_INTEGER,J,ierr)

      call MatGetType(J,matrixname,ierr)

! Store right-hand-side of PDE and exact solution
      call VecGetOwnershipRange(x,start,end,ierr)
      xp = h*start
      nn = end - start
      ii = start
      do 10, i=0,nn-1
        FF = 6.0*xp + (xp+1.e-12)**6.e0
        UU = xp*xp*xp
        call VecSetValues(ctx(2),i1,ii,FF,INSERT_VALUES,ierr)
        call VecSetValues(U,i1,ii,UU,INSERT_VALUES,ierr)
        xp = xp + h
        ii = ii + 1
 10   continue
      call VecAssemblyBegin(ctx(2),ierr)
      call VecAssemblyEnd(ctx(2),ierr)
      call VecAssemblyBegin(U,ierr)
      call VecAssemblyEnd(U,ierr)

! Create nonlinear solver
      call SNESCreate(PETSC_COMM_WORLD,snes,ierr)

! Set various routines and options
      call SNESSetFunction(snes,r,FormFunction,ctx,ierr)
      call SNESSetJacobian(snes,J,J,FormJacobian,ctx,ierr)
      call SNESSetFromOptions(snes,ierr)

! Solve nonlinear system
      call FormInitialGuess(snes,x,ierr)
      call SNESSolve(snes,PETSC_NULL_OBJECT,x,ierr)
      call SNESGetIterationNumber(snes,its,ierr);

! Write results if first processor
      if (ctx(4) .eq. 0) then
        write(6,100) its
      endif
  100 format('Number of Newton iterations = ',i5)

!  Free work space.  All PETSc objects should be destroyed when they
!  are no longer needed.
      call VecDestroy(x,ierr)
      call VecDestroy(ctx(3),ierr)
      call VecDestroy(r,ierr)
      call VecDestroy(U,ierr)
      call VecDestroy(ctx(2),ierr)
      call MatDestroy(J,ierr)
      call SNESDestroy(snes,ierr)
      call DADestroy(ctx(1),ierr)
      call PetscFinalize(ierr)
      end


! --------------------  Evaluate Function F(x) --------------------- 

      subroutine FormFunction(snes,x,f,ctx,ierr)
      implicit none
      SNES             snes
      Vec              x,f
      PetscFortranAddr ctx(*)
      PetscMPIInt  rank,size
      PetscInt i,s,n
      PetscErrorCode ierr
      PetscOffset      ixx,iff,iF2
      PetscScalar      h,d,vf2(1),vxx(1),vff(1)
#include "finclude/petscsys.h"
#include "finclude/petscvec.h"
#include "finclude/petscda.h"
#include "finclude/petscmat.h"
#include "finclude/petscsnes.h"


      rank  = ctx(4)
      size  = ctx(5)
      h     = 1.d0/(ctx(6) - 1.d0)
      call DAGlobalToLocalBegin(ctx(1),x,INSERT_VALUES,ctx(3),ierr)
      call DAGlobalToLocalEnd(ctx(1),x,INSERT_VALUES,ctx(3),ierr)

      call VecGetLocalSize(ctx(3),n,ierr)
      if (n .gt. 1000) then 
        print*, 'Local work array not big enough'
        call MPI_Abort(PETSC_COMM_WORLD,0,ierr)
      endif

!
! This sets the index ixx so that vxx(ixx+1) is the first local
! element in the vector indicated by ctx(3).
!
      call VecGetArray(ctx(3),vxx,ixx,ierr)
      call VecGetArray(f,vff,iff,ierr)
      call VecGetArray(ctx(2),vF2,iF2,ierr)

      d = h*h

!
!  Note that the array vxx() was obtained from a ghosted local vector 
!  ctx(3) while the array vff() was obtained from the non-ghosted parallel 
!  vector F. This is why there is a need for shift variable s. Since vff()
!  does not have locations for the ghost variables we need to index in it 
!  slightly different then indexing into vxx(). For example on processor 
!  1 (the second processor) 
!
!        xx(1)        xx(2)             xx(3)             .....
!      ^^^^^^^        ^^^^^             ^^^^^
!      ghost value   1st local value   2nd local value 
!
!                      ff(1)             ff(2)
!                     ^^^^^^^           ^^^^^^^
!                    1st local value   2nd local value
!
       if (rank .eq. 0) then 
        s = 0
        ff(1) = xx(1)
      else
        s = 1
      endif

      do 10 i=1,n-2
       ff(i-s+1) = d*(xx(i) - 2.d0*xx(i+1)                              &
     &      + xx(i+2)) + xx(i+1)*xx(i+1)                                &
     &      - F2(i-s+1)
 10   continue

      if (rank .eq. size-1) then
        ff(n-s) = xx(n) - 1.d0
      endif

      call VecRestoreArray(f,vff,iff,ierr)
      call VecRestoreArray(ctx(3),vxx,ixx,ierr)
      call VecRestoreArray(ctx(2),vF2,iF2,ierr)
      return
      end

! --------------------  Form initial approximation ----------------- 

      subroutine FormInitialGuess(snes,x,ierr)
      implicit none
#include "finclude/petscsys.h"
#include "finclude/petscvec.h"
#include "finclude/petscsnes.h"
      PetscErrorCode   ierr
      Vec              x
      SNES             snes
      PetscScalar      five 

      five = 5.d-1
      call VecSet(x,five,ierr)
      return
      end

! --------------------  Evaluate Jacobian -------------------- 

      subroutine FormJacobian(snes,x,jac,B,flag,ctx,ierr)
      implicit none
#include "finclude/petscsys.h"
#include "finclude/petscvec.h"
#include "finclude/petscda.h"
#include "finclude/petscmat.h"
#include "finclude/petscsnes.h"
      SNES             snes
      Vec              x
      Mat              jac,B
      PetscFortranAddr ctx(*)
      PetscTruth       flag
      PetscInt  ii,istart,iend
      PetscInt  i,j,n,end,start,i1
      PetscErrorCode ierr
      PetscMPIInt rank,size
      PetscOffset      ixx
      PetscScalar      d,A,h,vxx(1)

      i1 = 1
      h = 1.d0/(ctx(6) - 1.d0)
      d = h*h
      rank = ctx(4)
      size = ctx(5)

      call VecGetArray(x,vxx,ixx,ierr)
      call VecGetOwnershipRange(x,start,end,ierr)
      n = end - start

      if (rank .eq. 0) then
        A = 1.0 
        call MatSetValues(jac,i1,start,i1,start,A,INSERT_VALUES,ierr)
        istart = 1
      else 
        istart = 0
      endif
      if (rank .eq. size-1) then
        i = ctx(6)-1
        A = 1.0 
        call MatSetValues(jac,i1,i,i1,i,A,INSERT_VALUES,ierr)
        iend = n-1
      else
        iend = n
      endif
      do 10 i=istart,iend-1
        ii = i + start
        j = start + i - 1 
        call MatSetValues(jac,i1,ii,i1,j,d,INSERT_VALUES,ierr)
        j = start + i + 1 
        call MatSetValues(jac,i1,ii,i1,j,d,INSERT_VALUES,ierr)
        A = -2.0*d + 2.0*xx(i+1)
        call MatSetValues(jac,i1,ii,i1,ii,A,INSERT_VALUES,ierr)
 10   continue
      call VecRestoreArray(x,vxx,ixx,ierr)
      call MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY,ierr)
      return
      end



