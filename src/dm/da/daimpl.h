/* $Id: snes.h,v 1.17 1995/06/02 21:05:19 bsmith Exp $ */

/*

*/

#if !defined(_DAIMPL_H)
#define _DAIMPL_H
#include "ptscimpl.h"
#include "da.h"

struct _DA {
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
