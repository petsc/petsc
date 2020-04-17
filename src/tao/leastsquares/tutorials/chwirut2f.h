! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!             Include file for program chwirut2f.F
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
      PetscMPIInt  nn
      PetscMPIInt  rank
      PetscMPIInt  size
      PetscMPIInt  idle_tag, die_tag
      parameter (m=214)
      parameter (n=3)
      parameter (nn=n)
      parameter (idle_tag=2000)
      parameter (die_tag=3000)

      common /params/ t,y,rank,size

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
