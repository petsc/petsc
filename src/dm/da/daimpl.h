/* $Id: daimpl.h,v 1.7 1995/08/07 22:01:45 bsmith Exp bsmith $ */

/*

*/

#if !defined(_DAIMPL_H)
#define _DAIMPL_H
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
  int           wrap;              /* indicates if periodic boundaries */
  VecScatter    gtol,ltog;      
  Vec           global,local;
  DAStencilType stencil_type;
};

#endif
