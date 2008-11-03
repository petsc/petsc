
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

#include "fINCLUDE/petsc.h"
#include "fINCLUDE/is.h"
#include "fINCLUDE/vec.h"
#include "fINCLUDE/da.h"
#include "fINCLUDE/mat.h"
#include "fINCLUDE/ksp.h"
#include "fINCLUDE/pc.h"
#include "fINCLUDE/sles.h"
#include "fINCLUDE/snes.h"
#include "fINCLUDE/ts.h"
#include "fINCLUDE/viewer.h"
#include "fINCLUDE/draw.h"

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
C
C  The application context to contain data needed by the 
C  application-provided call-back routines, RHSFunction(),
C  RHSJacobian(), Monitor().  In this example the application context
C  is a Fortran common block, /appctx/.  Note that we can store
C  (pointers to) PETSc objects within this common block.
C    appctx:  dimension - problem dimension (1 or 2)
C             M         - total number of grid points in first dimension
C             N         - total number of grid points in second dimension
C             MN        - total number of grid points (including final
C                         unphysical points), where MN = M * N
C             da        - distributed array
C             solution  - solution vector
C             comm      - communicator
C             rank      - processor rank within communicator 
C                         (processor id from 0 to size-1)
C             size      - number of processors
C             debug     - flag (1 indicates debugging printouts)
C
C  Store other misc problem parameters in common block /params/
C             h         - mesh width h = 1/(M-1)
C
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
C  Common block data:
      DA               da
      Vec              solution
      integer          dimension, M, N, MN, rank, size, debug
      double precision zero_d0, one_d0, two_d0, four_d0
      MPI_Comm         comm
      Viewer           output

      common /params/ zero_d0, one_d0, two_d0, four_d0
      common /appctx/ dimension, M, N, MN, debug, da, solution
      common /appctx/ comm, rank, size, output

C Common block for local grid parameters
C       xs, ys   - local starting index
C       xe, ye   - local ending index
C       xm, ym   - local width
C       gxs, gys - local starting index (ghost)
C       gxe, gye - local ending index (ghost)
C       gxm, gym - local width (ghost)

      integer xs,xm,xe,gxs,gxm,gxe
      integer ys,ym,ye,gys,gym,gye

      common /gridp/ xs,xm,xe,gxs,gxm,gxe
      common /gridp/ ys,ym,ye,gys,gym,gye

C
C   Common block parameters for PDE parameters
C

      integer npar
      parameter (npar=11)
      double precision hi1,hi2,hi3,hi4,h,d1p1,d2p2,d3p2,dep,str
      double precision rpar(npar)

       common /invstep/hi1,hi2,hi3,hi4,h
       common /constep/d1p1,d2p2,d3p2
       common /coef/dep,str
       common /cbpar1/rpar

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
