!
!  $Id: param.h,v 1.21 1998/03/25 21:43:14 curfman Exp balay $;
!
! PETSc include files needed by Fortran routines
!   petsc.h - basic PETSc interface
!   mat.h   - matrices
!   vec.h   - vectors
!   ao.h    - application orderings
!
        implicit none

#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/mat.h"
#include "include/FINCLUDE/vec.h"
#include "include/FINCLUDE/ao.h"
#include "mmfort.h"

!   Parameters
        double precision one, two, zero, p5, pi
        parameter(zero=0.0d0,p5=0.5d0,one=1.0d0,two=2.0d0)

        parameter(pi=3.1415926578d0)

!   Type of system
        integer EXPLICIT, IMPLICIT_SIZE, IMPLICIT
        parameter(EXPLICIT=0, IMPLICIT_SIZE=1, IMPLICIT=2)

!   Type of scaling of system
        integer DT_MULT, DT_DIV
        parameter(DT_MULT=0, DT_DIV=1)

!   Type of timestepping
        integer LOCAL_TS, GLOBAL_TS
        parameter(LOCAL_TS=0, GLOBAL_TS=1)

!   Type of limiter
        integer LIM_NONE, LIM_MINMOD, LIM_SUPERBEE, LIM_VAN_LEER, 
     &          LIM_VAN_ALBADA
        parameter(LIM_NONE=0, LIM_MINMOD=1, LIM_SUPERBEE=2,
     &          LIM_VAN_LEER=3, LIM_VAN_ALBADA=4)

!   The following are the test problem dimensions:
       integer d_ni,d_nj,d_nk,d_ni1,d_nj1,d_nk1
!     needed for m6c
!       PARAMETER(D_NI=49,D_NJ=9,D_NK=9)
!     needed for m6f
!        PARAMETER(D_NI=97,D_NJ=17,D_NK=17)
!     needed for m6n
        PARAMETER(D_NI=193,D_NJ=33,D_NK=33)
        PARAMETER(D_NI1=D_NI+1,D_NJ1=D_NJ+1,D_NK1=D_NK+1)
!
! ------------------------------------------------------------
!   The following are the test problem dimensions:
       integer ni,nj,nk,ni1,nj1,nk1

!   Parallel implementation:
!
!   Ghost points, starting and ending values for each direction
!   Stencil width is 2
       integer gxsf, gysf, gzsf, gxef, gyef, gzef

!   Ghost points, starting and ending values for each direction
!   Stencil width is 1 (used for some work arrays)
       integer gxsfw, gysfw, gzsfw, gxefw, gyefw, gzefw
       integer gxsf2w, gysf2w, gzsf2w, gxsf1w, gysf1w, gzsf1w

!   Grid points: start, end, and width for each direction
       integer xsf, ysf, zsf, xef, yef, zef
       integer xef01, yef01, zef01, gxef01, gyef01, gzef01

!   Grid and ghost points corresponding to ni-1, nj-1, nk-1
!      (end and width for each direction)
       integer xefm1, yefm1, zefm1, xm, ym, zm
       integer gxm, gym, gzm
       integer gxefm1, gyefm1, gzefm1

!   Grid and ghost points corresponding to ni+1, nj+1, nk+1
!      (end and width for each direction)
       integer xefp1, yefp1, zefp1
       integer gxefp1, gyefp1, gzefp1

!   Grid points 2,2,2
       integer xsf2, ysf2, zsf2, gxsf2, gysf2, gzsf2

!   Grid points 1,1,1
       integer xsf1, ysf1, zsf1, gxsf1, gysf1, gzsf1

!   If nonzero, then print grid information
       integer printg, no_output

!   Deactivate wake wrap-around BCs in Jacobian if nonzero
       integer nowake

!   Type of boundary conditions, switch for impermeability
       integer bctype, bcswitch

!   Mesh boundaries (used in coord.h)
       integer cx1, cxn, cy1, cyn, cz1, czn

!   Problem number (1, 2, or 3), number of components per node
       integer problem, ndof, ndof_e

!   Communicator, rank, size
       integer comm, rank, size

!   Common block for local data
       common /pgrid/ rank, size, comm, problem, ndof, ndof_e
       common /pgrid/ printg, no_output, bctype, bcswitch
       common /pgrid/ nowake
       common /pgrid/ xsf, ysf, zsf, xef, yef, zef
       common /pgrid/ xsf2, ysf2, zsf2, xsf1, ysf1, zsf1
       common /pgrid/ xefm1, yefm1, zefm1, xm, ym, zm
       common /pgrid/ xefp1, yefp1, zefp1, xef01, yef01, zef01
       common /pgrid/ ni, nj, nk, ni1, nj1, nk1
       common /pgrid/ cx1, cxn, cy1, cyn, cz1, czn

!   Common block for local ghost parameters
       common /pghost/ gxsf, gysf, gzsf, gxef, gyef, gzef
       common /pghost/ gxm, gym, gzm
       common /pghost/ gxsf2, gysf2, gzsf2, gxsf1, gysf1, gzsf1
       common /pghost/ gxsfw, gysfw, gzsfw, gxefw, gyefw, gzefw
       common /pghost/ gxsf2w, gysf2w, gzsf2w, gxsf1w, gysf1w, gzsf1w
       common /pghost/ gxefp1, gyefp1, gzefp1
       common /pghost/ gxef01, gyef01, gzef01
       common /pghost/ gxefm1, gyefm1, gzefm1

!   Type of multi-model
        integer model

!   Boundaries for Euler part of multi-model:
        integer mm_xsf, mm_xef

        integer nk_boundary, nk1_boundary

!   Common block for multi-model data
       common /multimodel/ model, mm_xsf, mm_xef

!   Duct problem parameters
       double precision bump
       common /duct/ bump
       common /duct/ nk_boundary, nk1_boundary
