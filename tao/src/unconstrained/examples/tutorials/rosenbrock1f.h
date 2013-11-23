! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!             Include file for program rosenbrock1f.F
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!
!  This program uses CPP for preprocessing, as indicated by the use of
!  TAO include files in the directories $TAO_DIR/include/finclude and
!  $PETSC_DIR/include/finclude.  This convention enables use of the CPP
!  preprocessor, which allows the use of the #include statements that
!  define TAO objects and variables.
!
!  Since one must be very careful to include each file no more than once
!  in a Fortran routine, application programmers must explicitly list
!  each file needed for the various TAO and PETSc components within their
!  program (unlike the C/C++ interface).
!
!  See the Fortran section of the PETSc users manual for details.
!
!  The following include statements are generally used in TAO programs:
!     tao_solver.h - TAO solvers
!     petscksp.h   - Krylov subspace methods
!     petscpc.h    - preconditioners
!     petscmat.h   - matrices
!     petscvec.h   - vectors
!     petsc.h      - basic PETSc routines

#include "finclude/petscsys.h"
#include "finclude/petscvec.h"
#include "finclude/petscmat.h"
#include "finclude/petscksp.h"
#include "finclude/petscpc.h"
#include "finclude/taosolver.h"

!  Common blocks:
!  In this example we use common blocks to store data needed by the 
!  application-provided call-back routines, FormMinimizationFunction(),
!  FormFunctionGradient(), and FormHessian().  Note that we can store
!  (pointers to) TAO objects within these common blocks. 
!
!  common /params/ - contains parameters that help to define the application 
!
!     alpha, n - define the extended Rosenbrock function: 
!       sum_{i=0}^{n/2-1} ( alpha*(x_{2i+1}-x_{2i}^2)^2 + (1-x_{2i})^2 )

      PetscReal        alpha
      PetscInt         n

      common /params/ alpha, n

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
