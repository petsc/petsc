!
!
!  This example demonstrates basic use of the SNES Fortran interface.
!
!
        module UserModule
#include <petsc/finclude/petscsnes.h>
        use petscsnes
        type User
          DM  da
          Vec F
          Vec xl
          MPI_Comm comm
          PetscInt N
        end type User
        save
        type monctx
        PetscInt :: its,lag
        end type monctx
      end module

! ---------------------------------------------------------------------
! ---------------------------------------------------------------------
!  Subroutine FormMonitor
!  This function lets up keep track of the SNES progress at each step
!  In this routine, we determine when the Jacobian is rebuilt with the parameter 'jag'
!
!  Input Parameters:
!    snes    - SNES nonlinear solver context
!    its     - current nonlinear iteration, starting from a call of SNESSolve()
!    norm    - 2-norm of current residual (may be approximate)
!    snesm - monctx designed module (included in Snesmmod)
! ---------------------------------------------------------------------
      subroutine FormMonitor(snes,its,norm,snesm,ierr)
      use UserModule
      implicit none

      SNES ::           snes
      PetscInt ::       its,one,mone
      PetscScalar ::    norm
      type(monctx) ::   snesm
      PetscErrorCode :: ierr

!      write(6,*) ' '
!      write(6,*) '    its ',its,snesm%its,'lag',
!     &            snesm%lag
!      call flush(6)
      if (mod(snesm%its,snesm%lag).eq.0) then
        one = 1
        call SNESSetLagJacobian(snes,one,ierr)  ! build jacobian
      else
        mone = -1
        call SNESSetLagJacobian(snes,mone,ierr) ! do NOT build jacobian
      endif
      snesm%its = snesm%its + 1
      end subroutine FormMonitor

!  Note: Any user-defined Fortran routines (such as FormJacobian)
!  MUST be declared as external.
!
!
! Macros to make setting/getting  values into vector clearer.
! The element xx(ib) is the ibth element in the vector indicated by ctx%F
#define xx(ib)  vxx(ixx + (ib))
#define ff(ib)  vff(iff + (ib))
#define F2(ib)  vF2(iF2 + (ib))
      program main
      use UserModule
      implicit none
      type(User) ctx
      PetscMPIInt rank,size
      PetscErrorCode ierr
      PetscInt N,start,end,nn,i
      PetscInt ii,its,i1,i0,i3
      PetscBool  flg
      SNES             snes
      Mat              J
      Vec              x,r,u
      PetscScalar      xp,FF,UU,h
      character*(10)   matrixname
      external         FormJacobian,FormFunction
      external         formmonitor
      type(monctx) :: snesm

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif
      i1 = 1
      i0 = 0
      i3 = 3
      N  = 10
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,        &
     &                        '-n',N,flg,ierr)
      h = 1.0/real(N-1)
      ctx%N = N
      ctx%comm = PETSC_COMM_WORLD

      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)
      call MPI_Comm_size(PETSC_COMM_WORLD,size,ierr)

! Set up data structures
      call DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,N,i1,i1,            &
     &     PETSC_NULL_INTEGER,ctx%da,ierr)
      call DMSetFromOptions(ctx%da,ierr)
      call DMSetUp(ctx%da,ierr)
      call DMCreateGlobalVector(ctx%da,x,ierr)
      call DMCreateLocalVector(ctx%da,ctx%xl,ierr)

      call PetscObjectSetName(x,'Approximate Solution',ierr)
      call VecDuplicate(x,r,ierr)
      call VecDuplicate(x,ctx%F,ierr)
      call VecDuplicate(x,U,ierr)
      call PetscObjectSetName(U,'Exact Solution',ierr)

      call MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,N,           &
     &     N,i3,PETSC_NULL_INTEGER,i0,PETSC_NULL_INTEGER,J,ierr)
      call MatSetOption(J,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE,ierr)
      call MatGetType(J,matrixname,ierr)

! Store right-hand-side of PDE and exact solution
      call VecGetOwnershipRange(x,start,end,ierr)
      xp = h*start
      nn = end - start
      ii = start
      do 10, i=0,nn-1
        FF = 6.0*xp + (xp+1.e-12)**6.e0
        UU = xp*xp*xp
        call VecSetValues(ctx%F,i1,ii,FF,INSERT_VALUES,ierr)
        call VecSetValues(U,i1,ii,UU,INSERT_VALUES,ierr)
        xp = xp + h
        ii = ii + 1
 10   continue
      call VecAssemblyBegin(ctx%F,ierr)
      call VecAssemblyEnd(ctx%F,ierr)
      call VecAssemblyBegin(U,ierr)
      call VecAssemblyEnd(U,ierr)

! Create nonlinear solver
      call SNESCreate(PETSC_COMM_WORLD,snes,ierr)

! Set various routines and options
      call SNESSetFunction(snes,r,FormFunction,ctx,ierr)
      call SNESSetJacobian(snes,J,J,FormJacobian,ctx,ierr)

      snesm%its = 0
      call SNESGetLagJacobian(snes,snesm%lag,ierr)
      call SNESMonitorSet(snes,FormMonitor,snesm,                        &
     &        PETSC_NULL_FUNCTION,ierr)
      call SNESSetFromOptions(snes,ierr)

! Solve nonlinear system
      call FormInitialGuess(snes,x,ierr)
      call SNESSolve(snes,PETSC_NULL_VEC,x,ierr)
      call SNESGetIterationNumber(snes,its,ierr);

!  Free work space.  All PETSc objects should be destroyed when they
!  are no longer needed.
      call VecDestroy(x,ierr)
      call VecDestroy(ctx%xl,ierr)
      call VecDestroy(r,ierr)
      call VecDestroy(U,ierr)
      call VecDestroy(ctx%F,ierr)
      call MatDestroy(J,ierr)
      call SNESDestroy(snes,ierr)
      call DMDestroy(ctx%da,ierr)
      call PetscFinalize(ierr)
      end


! --------------------  Evaluate Function F(x) ---------------------

      subroutine FormFunction(snes,x,f,ctx,ierr)
      use UserModule
      implicit none
      SNES             snes
      Vec              x,f
      type(User) ctx
      PetscMPIInt  rank,size
      PetscInt i,s,n
      PetscErrorCode ierr
      PetscOffset      ixx,iff,iF2
      PetscScalar      h,d,vf2(2),vxx(2),vff(2)

      call MPI_Comm_rank(ctx%comm,rank,ierr)
      call MPI_Comm_size(ctx%comm,size,ierr)
      h     = 1.0/(real(ctx%N) - 1.0)
      call DMGlobalToLocalBegin(ctx%da,x,INSERT_VALUES,ctx%xl,ierr)
      call DMGlobalToLocalEnd(ctx%da,x,INSERT_VALUES,ctx%xl,ierr)

      call VecGetLocalSize(ctx%xl,n,ierr)
      if (n .gt. 1000) then
        print*, 'Local work array not big enough'
        call MPI_Abort(PETSC_COMM_WORLD,0,ierr)
      endif

!
! This sets the index ixx so that vxx(ixx+1) is the first local
! element in the vector indicated by ctx%xl.
!
      call VecGetArrayRead(ctx%xl,vxx,ixx,ierr)
      call VecGetArray(f,vff,iff,ierr)
      call VecGetArray(ctx%F,vF2,iF2,ierr)

      d = h*h

!
!  Note that the array vxx() was obtained from a ghosted local vector
!  ctx%xl while the array vff() was obtained from the non-ghosted parallel
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
       ff(i-s+1) = d*(xx(i) - 2.0*xx(i+1)                               &
     &      + xx(i+2)) + xx(i+1)*xx(i+1)                                &
     &      - F2(i-s+1)
 10   continue

      if (rank .eq. size-1) then
        ff(n-s) = xx(n) - 1.0
      endif

      call VecRestoreArray(f,vff,iff,ierr)
      call VecRestoreArrayRead(ctx%xl,vxx,ixx,ierr)
      call VecRestoreArray(ctx%F,vF2,iF2,ierr)
      return
      end

! --------------------  Form initial approximation -----------------

      subroutine FormInitialGuess(snes,x,ierr)
      use UserModule
      implicit none

      PetscErrorCode   ierr
      Vec              x
      SNES             snes
      PetscScalar      five

      five = .5
      call VecSet(x,five,ierr)
      return
      end

! --------------------  Evaluate Jacobian --------------------

      subroutine FormJacobian(snes,x,jac,B,ctx,ierr)
      use UserModule
      implicit none

      SNES             snes
      Vec              x
      Mat              jac,B
      type(User) ctx
      PetscInt  ii,istart,iend
      PetscInt  i,j,n,end,start,i1
      PetscErrorCode ierr
      PetscMPIInt rank,size
      PetscOffset      ixx
      PetscScalar      d,A,h,vxx(2)

      i1 = 1
      h = 1.0/(real(ctx%N) - 1.0)
      d = h*h
      call MPI_Comm_rank(ctx%comm,rank,ierr)
      call MPI_Comm_size(ctx%comm,size,ierr)

      call VecGetArrayRead(x,vxx,ixx,ierr)
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
        i = INT(ctx%N-1)
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
      call VecRestoreArrayRead(x,vxx,ixx,ierr)
      call MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY,ierr)
      return
      end

!/*TEST
!
!   test:
!      nsize: 2
!      args: -ksp_gmres_cgs_refinement_type refine_always -n 10 -snes_monitor_short
!      output_file: output/ex12_1.out
!
!TEST*/
