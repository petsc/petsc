c
c PETSc include files needed by Fortran routines
c   petsc.h - basic PETSc interface
c   mat.h   - matrices
c   vec.h   - vectors
c
        implicit none

#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/mat.h"
#include "include/FINCLUDE/vec.h"
#include "include/FINCLUDE/ao.h"

c   Parameters
        Double one, two, zero, p5
        parameter(zero=PetscDoubleExp(0.0,0),
     &              p5=PetscDoubleExp(0.5,0),
     &             one=PetscDoubleExp(1.0,0),
     &             two=PetscDoubleExp(2.0,0))

c   Type of system
        integer EXPLICIT, IMPLICIT_SIZE, IMPLICIT
        parameter(EXPLICIT=0, IMPLICIT_SIZE=1, IMPLICIT=2)

c   Type of scaling of system
        integer DT_MULT, DT_DIV
        parameter(DT_MULT=0, DT_DIV=1)

c   Type of timestepping
        integer LOCAL_TS, GLOBAL_TS
        parameter(LOCAL_TS=0, GLOBAL_TS=1)

c   The following are the test problem dimensions:
       integer d_ni,d_nj,d_nk,d_ni1,d_nj1,d_nk1
c     needed for m6c
c       PARAMETER(D_NI=49,D_NJ=9,D_NK=9)
c     needed for m6f
c        PARAMETER(D_NI=97,D_NJ=17,D_NK=17)
c     needed for m6n
        PARAMETER(D_NI=193,D_NJ=33,D_NK=33)
        PARAMETER(D_NI1=D_NI+1,D_NJ1=D_NJ+1,D_NK1=D_NK+1)
c
c ------------------------------------------------------------
c   The following are the test problem dimensions:
       integer ni,nj,nk,ni1,nj1,nk1

c   Parallel implementation:
c
c   Ghost points, starting and ending values for each direction
c   Stencil width is 2
       integer gxsf, gysf, gzsf, gxef, gyef, gzef

c   Ghost points, starting and ending values for each direction
c   Stencil width is 1 (used for some work arrays)
       integer gxsfw, gysfw, gzsfw, gxefw, gyefw, gzefw
       integer gxsf2w, gysf2w, gzsf2w, gxsf1w, gysf1w, gzsf1w

c   Grid points: start, end, and width for each direction
       integer xsf, ysf, zsf, xef, yef, zef
       integer xef01, yef01, zef01, gxef01, gyef01, gzef01

c   Grid and ghost points corresponding to ni-1, nj-1, nk-1
c      (end and width for each direction)
       integer xefm1, yefm1, zefm1, xm, ym, zm
       integer gxm, gym, gzm

c   Grid and ghost points corresponding to ni+1, nj+1, nk+1
c      (end and width for each direction)
       integer xefp1, yefp1, zefp1
       integer gxefp1, gyefp1, gzefp1

c   Grid points 2,2,2
       integer xsf2, ysf2, zsf2, gxsf2, gysf2, gzsf2

c   Grid points 1,1,1
       integer xsf1, ysf1, zsf1, gxsf1, gysf1, gzsf1

c   If nonzero, then print grid information
       integer printg, no_output

c   Type of boundary conditions
       integer bctype

c   Problem number (1, 2, or 3), number of components per node
       integer problem, nc

c   Communicator, rank, size
       integer comm, rank, size

c   Common block for local data
       common /pgrid/ rank, size, comm, problem, nc
       common /pgrid/ printg, no_output, bctype
       common /pgrid/ xsf, ysf, zsf, xef, yef, zef
       common /pgrid/ xsf2, ysf2, zsf2, xsf1, ysf1, zsf1
       common /pgrid/ xefm1, yefm1, zefm1, xm, ym, zm
       common /pgrid/ xefp1, yefp1, zefp1, xef01, yef01, zef01
       common /pgrid/ ni, nj, nk, ni1, nj1, nk1

c   Common block for local ghost parameters
       common /pghost/ gxsf, gysf, gzsf, gxef, gyef, gzef
       common /pghost/ gxm, gym, gzm
       common /pghost/ gxsf2, gysf2, gzsf2, gxsf1, gysf1, gzsf1
       common /pghost/ gxsfw, gysfw, gzsfw, gxefw, gyefw, gzefw
       common /pghost/ gxsf2w, gysf2w, gzsf2w, gxsf1w, gysf1w, gzsf1w
       common /pghost/ gxefp1, gyefp1, gzefp1
       common /pghost/ gxef01, gyef01, gzef01

c
