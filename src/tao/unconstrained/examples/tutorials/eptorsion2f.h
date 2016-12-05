! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!             Include file for program eptorsion2f.F
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  See the Fortran section of the PETSc users manual for details.
!

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
!     param      - nonlinearity parameter
!
!  common /pdata/ - contains some parallel data
!     localX     - local work vector (including ghost points)
!     localS     - local work vector (including ghost points)
!     dm         - distributed array
!
      Vec              localX
      DM               dm
      PetscReal      param
      PetscInt         mx, my

      common /params/ param,mx,my
      common /pdata/  dm,localX

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


