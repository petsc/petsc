!  This file contains include statements and a user-defined
!  common block for application-specific data.  This file is
!  included in each routine within the program ex2f. 
!
!  The following include statements are generally used in TS Fortran
!  programs:
!     petscsys.h       - base PETSc routines
!     petscvec.h    - vectors
!     petscmat.h    - matrices
!     petscksp.h    - Krylov subspace methods
!     petscpc.h     - preconditioners
!     petscsnes.h   - SNES interface
!     petscts.h     - TS interface
!     petscviewer.h - viewers
!     petscdraw.h   - drawing
!  In addition, we need the following for use of distributed arrays
!     petscda.h     - distributed arrays (DAs)
!  Other include statements may be needed if using additional PETSc
!  routines in a Fortran program, e.g.,
!     petscis.h     - index sets

#include "finclude/petscsys.h"
#include "finclude/petscvec.h"
#include "finclude/petscda.h"
#include "finclude/petscmat.h"
#include "finclude/petscksp.h"
#include "finclude/petscpc.h"
#include "finclude/petscsnes.h"
#include "finclude/petscts.h"
#include "finclude/petscviewer.h"
#include "finclude/petscdraw.h"

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!
!  The application context to contain data needed by the 
!  application-provided call-back routines, RHSFunction(),
!  RHSJacobian(), Monitor().  In this example the application context
!  is a Fortran common block, /appctx/.  Note that we can store
!  (pointers to) PETSc objects within this common block.
!    appctx:  M         - total number of grid points  
!             da        - distributed array
!             localwork - local work vector (including ghost points)
!             solution  - solution vector
!             comm      - communicator
!             rank      - processor rank within communicator
!             size      - number of processors
!             debug     - flag (1 indicates debugging printouts)
!
!  Store other misc problem parameters in common block /params/
!             h         - mesh width h = 1/(M-1)
!
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!  Common block data:
      DA    da
      Vec   localwork,solution
      Vec   u_local
      PetscInt   M
      PetscTruth debug
      PetscMPIInt size,rank
      PetscReal  zero_d0,one_d0
      PetscReal  two_d0,four_d0
      MPI_Comm   comm
      PetscReal  h

      common /params/ zero_d0,one_d0,two_d0,four_d0,h
      common /appctx/ localwork,solution,da,u_local
      common /appctx/ comm,rank,size,debug,M

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
