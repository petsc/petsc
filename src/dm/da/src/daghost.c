#ifndef lint
static char vcid[] = "$Id: daghost.c,v 1.1 1996/01/30 04:28:02 bsmith Exp bsmith $";
#endif
 
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "daimpl.h"    /*I   "da.h"   I*/

/*@
    DAGetGhostCorners - Returns the global (x,y,z) indices of the lower left
    corner of the local region, including ghost points.

   Input Parameter:
.  da - the distributed array

   Output Parameters:
.  x,y,z - the corner indices. y and z are optional.
.  m,n,p - widths in the corresponding directions. n and p are optional.

.keywords: distributed array, get, ghost, corners, nodes, local indices

.seealso: DAGetCorners()
@*/
int DAGetGhostCorners(DA da,int *x,int *y,int *z,int *m, int *n, int *p)
{
  int w;

  PETSCVALIDHEADERSPECIFIC(da,DA_COOKIE);
  /* since the xs, xe ... have all been multiplied by the number of degrees 
     of freedom per cell, w = da->w, we divide that out before returning.*/
  w = da->w;  
  *x = da->Xs/w; *m = (da->Xe - da->Xs)/w;

  if (y) *y = da->Ys; if (n) *n = (da->Ye - da->Ys);
  if (z) *z = da->Zs; if (p) *p = (da->Ze - da->Zs); 
  return 0;
}

