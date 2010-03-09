!
      module mex34f90
#include "finclude/petsc.h90"

!   Data structure used to contain information about the problem
!   You can add physical values etc here

      type appctx
        MPI_Comm :: comm = MPI_COMM_WORLD
        integer :: nxc = 5    ! number of grid points in channel
        integer :: np = 2,nc = 3  ! number of unknowns in pool and channel
      end type appctx

!    The names of the fields in the pool and in the channel
!    These are accessed via  variablename%p0, variablename%p1
!    change these to names appropriate for your physics

      type poolfield
        double precision :: p0,p1   ! unknowns in pool
      end type poolfield

      type channelfield
        double precision :: c0,c1,c2   ! unknowns in channel
      end type channelfield

      end module mex34f90

!
!   These are interface definitions that allow PETSc routines to be
!   called with "nice" names from Fortran90.
!
!   You should not need to change these, someday I hope to be able
!   to no longer require them
!
#define USERMODULE mex34f90
#define USERFIELD1 channelfield
#define USERFIELD2 poolfield

      module mex34finterfaces
          use mex34f90
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

      end module mex34finterfaces


