#define PETSC_AVOID_DECLARATIONS
#include "include/finclude/petscall.h"

      program ex34f
      use mex34f


      DMMG             dmmg
      PetscErrorCode   ierr
      DA               da
      DM               dm
      type(appctx)     app
      external         FormInitialGuess,FormFunction


      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

!      Unknowns consist of a one-dimensional field, a zero dimensional pool
!          another one-dimensional field and a final zero dimensional pool

      call DMCompositeCreate(PETSC_COMM_WORLD,dm,ierr)
      call DACreate1d(PETSC_COMM_WORLD,DA_NONPERIODIC,app%nxc,app%nc,1, &
     &                PETSC_NULL_INTEGER,da,ierr)
      call DMCompositeAddDA(dm,da,ierr)
      call DMCompositeAddArray(dm,0,app%np,ierr)
      call DMCompositeAddDA(dm,da,ierr)
      call DMCompositeAddArray(dm,0,app%np,ierr)
      call DADestroy(da,ierr)

!       Create solver object and associate it with the unknowns

      call DMMGCreate(PETSC_COMM_WORLD,1,PETSC_NULL_OBJECT,dmmg,ierr)
      call DMMGSetDM(dmmg,dm,ierr)
      call DMCompositeDestroy(dm,ierr)
      call DMMGSetUser(dmmg,0,app,ierr)  ! currently only one level
      call DMMGSetInitialGuess(dmmg,FormInitialGuess,ierr)
      call DMMGSetSNES(dmmg,FormFunction,PETSC_NULL_OBJECT,ierr)

      call DMMGSolve(dmmg,ierr)

      call DMMGDestroy(dmmg,ierr)
      call PetscFinalize(ierr)
      end


      subroutine FormInitialGuess(dmmg,v,ierr)
      use mex34f
      use mex34finterfaces

      DMMG dmmg
      DMComposite dm
      Vec  v,vc1,vc2
      PetscErrorCode ierr
      PetscInt np,i
      DA da
      type(DALocalInfof90) dainfo
      type(poolfield), pointer :: Pool1,Pool2
      type(channelfield), pointer :: v1(:),v2(:)
      type(appctx), pointer :: app
      PetscMPIInt rank

      call DMMGGetUser(dmmg,app,ierr)   ! access user context

      call DMMGGetDM(dmmg,dm,ierr)
      call DMCompositeGetEntries(dm,da,np,da,np,ierr) ! get the da's and sizes that define the unknowns
      call DAGetLocalInfof90(da,dainfo,ierr)

      call DMCompositeGetAccess(dm,v,vc1,Pool1,vc2,Pool2,ierr)  ! acess the unknowns in global (nonghosted) form
      call DAVecGetArrayf90(da,vc1,v1,ierr)
      call DAVecGetArrayf90(da,vc2,v2,ierr)

      call MPI_Comm_rank(app%comm,rank,ierr)
      if (rank == 0) then
         Pool1%p0 = -2.0
         Pool1%p1 = -2000.0
         Pool2%p0 = -20
         Pool2%p1 = -200
      endif

      do i=dainfo%xs,dainfo%xs+dainfo%xm-1
        v1(i)%c0 = 3*i - .5
        v1(i)%c1 = 3*i - (1.d0/3.d0)
        v1(i)%c2 = 3*i - .1
      enddo 

      call DAVecRestoreArrayf90(da,vc1,v1,ierr)
      call DAVecRestoreArrayf90(da,vc2,v2,ierr)
      call DMCompositeRestoreAccess(dm,v,vc1,Pool1,vc2,Pool2,ierr)

      CHKMEMQ
      call VecView(v,PETSC_NULL_OBJECT,ierr)
      return
      end      

      subroutine FormFunction(snes,x,f,ierr)
      use mex34f
      use mex34finterfaces

      Vec  x,f
      PetscErrorCode ierr
      SNES snes

      call VecCopy(x,f,ierr)
      return
      end      

