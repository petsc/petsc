#ifndef lint
static char vcid[] = "$Id: dacorn.c,v 1.4 1996/03/19 21:29:33 bsmith Exp bsmith $";
#endif
 
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "src/da/daimpl.h"    /*I   "da.h"   I*/

/*@
   DAGetCorners - Returns the global (x,y,z) indices of the lower left
   corner of the local region, excluding ghost points.

   Input Parameter:
.  da - the distributed array

   Output Parameters:
.  x,y,z - the corner indices
$    y and z are optional (used for 2D and 3D problems)
.  m,n,p - widths in the corresponding directions
$    n and p are optional (used for 2D and 3D problems)

   Note:
   Any of y, z, n, and p should be set to PETSC_NULL if not needed.

.keywords: distributed array, get, corners, nodes, local indices

.seealso: DAGetGhostCorners()
@*/
int DAGetCorners(DA da,int *x,int *y,int *z,int *m, int *n, int *p)
{
  int w;

  PetscValidHeaderSpecific(da,DA_COOKIE);
  /* since the xs, xe ... have all been multiplied by the number of degrees 
     of freedom per cell, w = da->w, we divide that out before returning.*/
  w = da->w;  
  *x = da->xs/w; *m = (da->xe - da->xs)/w;
  /* the y and z have NOT been multiplied by w */
  if (y) *y = da->ys; if (n) *n = (da->ye - da->ys);
  if (z) *z = da->zs; if (p) *p = (da->ze - da->zs); 
  return 0;
} 

