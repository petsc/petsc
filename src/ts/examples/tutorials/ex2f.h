!  This file contains include statements and a user-defined
!  common block for application-specific data.  This file is
!  included in each routine within the program ex2f. 
!
!  The following include statements are generally used in TS Fortran
!  programs:
!     petsc.h  - base PETSc routines
!     vec.h    - vectors
!     mat.h    - matrices
!     ksp.h    - Krylov subspace methods
!     pc.h     - preconditioners
!     sles.h   - SLES interface
!     snes.h   - SNES interface
!     ts.h     - TS interface
!     viewer.h - viewers
!     draw.h   - drawing
!  In addition, we need the following for use of distributed arrays
!     da.h     - distributed arrays (DAs)
!  Other include statements may be needed if using additional PETSc
!  routines in a Fortran program, e.g.,
!     is.h     - index sets

#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"
#include "include/FINCLUDE/da.h"
#include "include/FINCLUDE/mat.h"
#include "include/FINCLUDE/ksp.h"
#include "include/FINCLUDE/pc.h"
#include "include/FINCLUDE/sles.h"
#include "include/FINCLUDE/snes.h"
#include "include/FINCLUDE/ts.h"
#include "include/FINCLUDE/viewer.h"
#include "include/FINCLUDE/draw.h"

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
      DA               da
      Vec              localwork, solution
      integer          M, rank, size, debug
      double precision h, zero_d0, one_d0, two_d0, four_d0
      MPI_Comm         comm

      common /params/ h, zero_d0, one_d0, two_d0, four_d0
      common /appctx/ M, debug, da, localwork, solution
      common /appctx/ comm, rank, size

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
