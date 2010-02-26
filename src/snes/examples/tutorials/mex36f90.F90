!
      module mex36f90
#include "finclude/petsc.h90"

!   Data structure used to contain information about the problem
!   You can add physical values etc here

      type appctx
        MPI_Comm :: comm = MPI_COMM_WORLD
        integer :: nxc = 5         ! number of grid points in channel
        integer :: np = 2,nc = 1   ! number of unknowns in pool and channel
        double precision :: P0     ! atmospheric pressure
        double precision :: rho    ! fluid density
        double precision :: grav   ! gravity
        double precision :: dhhpl0 ! initial height of hot pool level
        double precision :: dhcpl0 ! initial height of cold pool level
        double precision :: dhci   ! height of core inlet
        double precision :: dhco   ! height of core outlet
        double precision :: dhii   ! height of IHX inlet
        double precision :: dhio   ! height of IHX outlet
        double precision :: lenc   ! core length
        double precision :: leni   ! IHX length
        double precision :: dxc    ! mesh spacing in core
        double precision :: dxi    ! mesh spacing in IHX
        double precision :: dt     ! time step size
        integer :: nstep = 5       ! number of time steps
        double precision :: hpvelo ! old time hot pool velocity
        double precision :: hpvolo ! old time hot pool volume
        double precision :: cpvelo ! old time cold pool velocity
        double precision :: cpvolo ! old time cold pool volume
        double precision :: hpvol0 ! initial hot pool volume
        double precision :: cpvol0 ! initial cold pool volume
        double precision :: ahp    ! area of the hot pool
        double precision :: acp    ! area of the cold pool
        double precision :: acore  ! area of the core
        double precision :: aihx   ! area of the ihx
        Vec :: xold   ! old time state variables
      end type appctx

!    The names of the fields in the pool and in the channel
!    These are accessed via  variablename%p0, variablename%p1
!    change these to names appropriate for your physics

      type poolfield
        double precision :: vel,vol   ! unknowns in pool
      end type poolfield

      type channelfield
        double precision :: press   ! unknowns in channel
      end type channelfield

!     Stores all the local (ghosted) variables together 
!     for easy access

      type LocalForm
        PetscInt np
        DA  da
        type(poolfield), pointer :: HotPool,ColdPool
        type(channelfield), pointer :: IHX(:),Core(:)
        type(DALocalInfof90) dainfo
        Vec vIHX,vCore
      end type LocalForm

      end module mex36f90

!
!   These are interface definitions that allow PETSc routines to be
!   called with "nice" names from Fortran90.
!
!   You should not need to change these, someday I hope to be able
!   to no longer require them
!
#define USERMODULE mex36f90
#define USERFIELD1 channelfield
#define USERFIELD2 poolfield

      module mex36f90interfaces
          use mex36f90
      Interface DAVecGetArrayF90
        Subroutine DAVecGetArrayF90user1(Da, v,d1,ierr)
          use USERMODULE
          DA  da
          Vec v
          type(USERFIELD1), pointer :: d1(:)
          PetscErrorCode ierr
        End Subroutine
      End Interface DAVecGetArrayF90

      interface DAVecRestoreArrayF90
        Subroutine DAVecRestoreArrayF90user1(Da, v,d1,ierr)
          use USERMODULE
          DA  da
          Vec v
          type(USERFIELD1), pointer :: d1(:)
          PetscErrorCode ierr
        End Subroutine
      End Interface DAVecRestoreArrayF90

      interface DMMGSetUser
        Subroutine DMMGSetUser(dmmg, level,app,ierr)
          use USERMODULE
          DMMG dmmg
          type(appctx), pointer :: app
          PetscErrorCode ierr
          integer level
        End Subroutine
      End Interface DMMGSetUser

      interface DMMGGetUser
        Subroutine DMMGGetUser(dmmg, app,ierr)
          use USERMODULE
          DM dmmg
          type(appctx), pointer :: app
          PetscErrorCode ierr
        End Subroutine
      End Interface DMMGGetUser

      Interface DMCompositeGetAccess
        Subroutine DMCompositeGetAccess4(dm, v,d1,d2,d3,d4,ierr)
          use USERMODULE
          DM  dm
          Vec v,d1,d3
          type(poolfield),pointer :: d2,d4
          PetscErrorCode ierr
        End Subroutine
      End Interface

      Interface DMCompositeRestoreAccess
        Subroutine DMCompositeRestoreAccess4(dm, v,d1,d2,d3,d4,ierr)
          use USERMODULE
          DMComposite  dm
          Vec v,d1,d3
          type(poolfield),pointer :: d2,d4
          PetscErrorCode ierr
        End Subroutine
      End Interface

      Interface DMCompositeGetLocalVectors
        Subroutine DMCompositeGetLocalVectors4(dm, d1,p1,d2,p2,ierr)
          use USERMODULE
          DMComposite  dm
          type(poolfield),pointer :: p1,p2
          Vec d1,d2
          PetscErrorCode ierr
        End Subroutine
      End Interface

      Interface DMCompositeRestoreLocalVectors
        Subroutine DMCompositeRestoreLocalVectors4(dm, d1,p1,d2,p2,ierr)
          use USERMODULE
          DMComposite  dm
          type(poolfield),pointer :: p1,p2
          Vec d1,d2
          PetscErrorCode ierr
        End Subroutine
      End Interface

      Interface DMCompositeScatter
        Subroutine DMCompositeScatter4(dm, v,d1,d2,d3,d4,ierr)
          use USERMODULE
          DM  dm
          Vec v,d1,d3
          type(poolfield) d2,d4
          PetscErrorCode ierr
        End Subroutine
      End Interface

      end module mex36f90interfaces


