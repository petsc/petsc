/* $Id: daimpl.h,v 1.24 1999/03/01 04:58:24 bsmith Exp bsmith $ */

/*
   Distributed arrays - communication tools for parallel, rectangular grids.
*/

#if !defined(_DAIMPL_H)
#define _DAIMPL_H
#include "da.h"
#include "dfvec.h"

struct _p_DA {
  PETSCHEADER(int)
  int            (*destroy)(DA);
  int            (*view)(DA,Viewer);    

  int            M,N,P;                 /* array dimensions */
  int            m,n,p;                 /* processor layout */
  int            w;                     /* degrees of freedom per node */
  int            s;                     /* stencil width */
  int            xs,xe,ys,ye,zs,ze;     /* range of local values */
  int            Xs,Xe,Ys,Ye,Zs,Ze;     /* range including ghost values */
                                        /* values above already scaled by w */
  int            *idx,Nl;               /* local to global map */
  int            base;                  /* global number of 1st local node */
  DAPeriodicType wrap;                  /* indicates type of periodic boundaries */
  VecScatter     gtol, ltog, ltol;      /* scatters, see below for details */
  DFVec          global, local;         /* vectors that are discrete functions */
  PetscTruth     globalused, localused; /* indicates if global and local have been 
                                           passed back to user via DACreateXXXVector */
  DAStencilType  stencil_type;          /* stencil, either box or star */
  int            dim;                   /* DA dimension (1,2, or 3) */
  int            *gtog1;                /* mapping from global ordering to
                                            ordering that would be used for 1
                                            proc; intended for internal use only */
  DF             dfshell;               /* discrete function shell */
  AO             ao;                    /* application ordering context */

  ISLocalToGlobalMapping ltogmap;       /* local to global mapping for associated vectors */
  Vec                    coordinates;   /* coordinates (x,y,x) of local nodes, not including ghosts*/
  char                   **fieldname;   /* names of individual components in vectors */

  int                    *lx,*ly,*lz;   /* number of nodes in each partition block along 3 axis */
  Vec                    natural;       /* global vector for storing items in natural order */
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
            nearest neighbor timestepping.
*/

extern int DFShellCreateDA_Private(MPI_Comm,char**,DA,DF*);
extern int DAGetGlobalToGlobal1_Private(DA,int**);

#endif
