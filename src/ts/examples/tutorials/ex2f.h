
C  This file contains include statements and a user-defined
C  common block for application-specific data.  This file is
C  included in each routine within the program ex2f. 
C
C  The following include statements are generally used in TS Fortran
C  programs:
C     petsc.h  - base PETSc routines
C     vec.h    - vectors
C     mat.h    - matrices
C     ksp.h    - Krylov subspace methods
C     pc.h     - preconditioners
C     sles.h   - SLES interface
C     snes.h   - SNES interface
C     ts.h     - TS interface
C     viewer.h - viewers
C     draw.h   - drawing
C  In addition, we need the following for use of distributed arrays
C     da.h     - distributed arrays (DAs)
C  Other include statements may be needed if using additional PETSc
C  routines in a Fortran program, e.g.,
C     is.h     - index sets

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

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
C
C  The application context to contain data needed by the 
C  application-provided call-back routines, RHSFunction(),
C  RHSJacobian(), Monitor().  In this example the application context
C  is a Fortran common block, /appctx/.  Note that we can store
C  (pointers to) PETSc objects within this common block.
C    appctx:  M         - total number of grid points  
C             da        - distributed array
C             localwork - local work vector (including ghost points)
C             solution  - solution vector
C             comm      - communicator
C             rank      - processor rank within communicator
C             size      - number of processors
C             debug     - flag (1 indicates debugging printouts)
C
C  Store other misc problem parameters in common block /params/
C             h         - mesh width h = 1/(M-1)
C
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
C  Common block data:
      DA       da
      Vec      localwork, solution
      integer  M, rank, size, debug
      Double   h, zero_d0, one_d0, two_d0, four_d0
      MPI_Comm comm

      common /params/ h, zero_d0, one_d0, two_d0, four_d0
      common /appctx/ M, debug, da, localwork, solution
      common /appctx/ comm, rank, size

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
