/* $Id: daimpl.h,v 1.11 1996/02/17 16:35:35 curfman Exp curfman $ */

/*
   Distributed arrays - communication tools for parallel, rectangular grids.
*/

#if !defined(_DAIMPL_H)
#define _DAIMPL_H
#include "da.h"

struct _DA {
  PETSCHEADER
  int            M,N,P;             /* array dimensions */
  int            m,n,p;             /* processor layout */
  int            w;                 /* degrees of freedom per node */
  int            s;                 /* stencil width */
  int            xs,xe,ys,ye,zs,ze; /* range of local values */
  int            Xs,Xe,Ys,Ye,Zs,Ze; /* range including ghost values */
  int            *idx,Nl;           /* local to global map */
  int            base;              /* global number of 1st local node */
  DAPeriodicType wrap;              /* indicates type of periodic boundaries */
  VecScatter     gtol, ltog, ltol;  /* scatters, see below for details */
  Vec            global, local;     /* vectors */
  DAStencilType  stencil_type;      /* stencil, either box or star */
  int            dim;               /* DA dimension (1,2, or 3) */
  int            *gtog1;            /* mapping from global ordering to
                                       ordering that would be used for 1
                                       proc; intended for internal use only */
};

/*
  Vectors:
     Global has on each processor the interior degrees of freedom and
         no ghost points. This vector is what the solvers usually see.
     Local has on each processor the ghost points as well. This is 
          what code to calculate Jacobians, etc. usually sees.
  Vector scatters:
     gtol - Global representation to local
     ltog - Local representation to global (involves no communication)
     ltol - Local representation to local representation, updates the
            ghostpoint values in the second vector from (correct) interior
            values in the first vector.  This is good for explicit
            nearest neighbor time-stepping.
*/
#endif
