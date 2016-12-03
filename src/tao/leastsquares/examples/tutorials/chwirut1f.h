! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!             Include file for program chwirut1f.F
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!

#include "petsc/finclude/petsctao.h"
      use petsctao
      implicit none

!  Common blocks:
!  In this example we use common blocks to store data needed by the
!  application-provided call-back routines, FormMinimizationFunction(),
!  FormFunctionGradient(), and FormHessian().  Note that we can store
!  (pointers to) TAO objects within these common blocks.
!
!  common /params/ - contains parameters that help to define the application
!
      PetscReal t(0:213)
      PetscReal y(0:213)
      PetscInt  m,n

      common /params/ t,y,m,n

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
