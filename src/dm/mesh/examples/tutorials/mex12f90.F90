!
      module mex12f90
#include "include/finclude/petscall.h90"

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

      end module mex12f90



