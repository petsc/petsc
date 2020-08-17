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
!     alpha, n - define the extended Rosenbrock function:
!       sum_{i=0}^{n/2-1} ( alpha*(x_{2i+1}-x_{2i}^2)^2 + (1-x_{2i})^2)

      PetscReal        alpha
      PetscInt         n

      common /params/ alpha, n

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
