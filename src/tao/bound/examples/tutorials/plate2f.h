! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!             Include file for program plate.f
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
#include "petsc/finclude/petscdmda.h"
#include "petsc/finclude/petsctao.h"
      use petscdmda
      use petsctao
      implicit none

!  Common blocks:
!  In this example we use common blocks to store data needed by the
!  application-provided call-back routines, FormFunction(), FormGradient(),
!  and FormHessian().  Note that we can store (pointers to) TAO objects
!  within these common blocks.
!
!  common /params/ - contains parameters for the global application
!     mx, my     - global discretization in x- and y-directions
!     hx, hy     - mesh spacing in x- and y-directions
!     N          - dimension of global vectorn
!     bheight    - height of plate
!     bmx,bmy    - grid dimensions under plate
!
!  common /pdata/ - contains some parallel data
!     localX     - local work vector (including ghost points)
!     localV     - local work vector (including ghost points)
!     Top, Bottom, Left, Right - boundary vectors
!     Nx, Ny     - number of processes in x- and y- directions
!     dm         - distributed array

      Vec              localX, localV
      Vec              Top, Left
      Vec              Right, Bottom
      DM               dm
      PetscReal      bheight
      PetscInt         bmx, bmy
      PetscInt         mx, my, Nx, Ny, N


      common /params/ mx,my,bmx,bmy,bheight,N
      common /pdata/  dm,localX,localV,Nx,Ny
      common /pdata/  Left, Top, Right, Bottom

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


