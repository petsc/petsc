!
!  See the Fortran section of the PETSc users manual for details.
!
!  Common blocks:
!  In this example we use common blocks to store data needed by the
!  application-provided call-back routines, FormJacobian() and
!  FormFunction().  Note that we can store (pointers to)
!  PETSc objects within these common blocks.
!
!  common /params/ - contains parameters for the global application
!     mx, my   - global discretization in x- and y-directions
!     lambda   - nonlinearity parameter
!
!  common /pdata/  - contains some parallel data
!     da       - distributed array
!     rank     - processor rank within communicator
!     size     - number of processors
!     xs, ys   - local starting grid indices (no ghost points)
!     xm, ym   - widths of local grid (no ghost points)
!     gxs, gys - local starting grid indices (including ghost points)
!     gxm, gym - widths of local grid (including ghost points)

      PetscInt xs,xe,xm,gxs,gxe,gxm
      PetscInt ys,ye,ym,gys,gye,gym
      PetscInt mx,my
      PetscMPIInt rank,size
      PetscReal lambda

      common /params/ lambda,mx,my
      common /pdata/  xs,xe,xm,gxs,gxe,gxm
      common /pdata/  ys,ye,ym,gys,gye,gym
      common /pdata/  rank,size

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
