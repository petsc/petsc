
/*

*/

#if !defined(_RAIMPL_H)
#define _RAIMPL_H
#include "ptscimpl.h"
#include "ra.h"

struct _RA {
  PETSCHEADER
  int           M,N,P;             /* array dimensions */
  int           m,n,p;             /* processor layout */
  int           w;                 /* degrees of freedome per node */
  int           s;                 /* stencil width */
  int           xs,xe,ys,ye,zs,ze; /* range of local values */
  int           Xs,Xe,Ys,Ye,Zs,Ze; /* range including ghost values*/
  int           *idx,Nl;           /* local to global map */
  int           base;              /* global number of 1st local node */
  VecScatterCtx gtol,ltog;      
  Vec           global,local;
};

#endif
