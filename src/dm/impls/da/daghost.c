
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include <petsc/private/dmdaimpl.h>    /*I   "petscdmda.h"   I*/

/*@C
   DMDAGetGhostCorners - Returns the global (x,y,z) indices of the lower left
   corner and size of the local region, including ghost points.

   Not Collective

   Input Parameter:
.  da - the distributed array

   Output Parameters:
+  x,y,z - the corner indices (where y and z are optional; these are used
           for 2D and 3D problems)
-  m,n,p - widths in the corresponding directions (where n and p are optional;
           these are used for 2D and 3D problems)

   Level: beginner

   Note:
   The corner information is independent of the number of degrees of
   freedom per node set with the DMDACreateXX() routine. Thus the x, y, z, and
   m, n, p can be thought of as coordinates on a logical grid, where each
   grid point has (potentially) several degrees of freedom.
   Any of y, z, n, and p can be passed in as NULL if not needed.

.seealso: DMDAGetCorners(), DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMDAGetOwnershipRanges(), DMStagGetGhostCorners()

@*/
PetscErrorCode  DMDAGetGhostCorners(DM da,PetscInt *x,PetscInt *y,PetscInt *z,PetscInt *m,PetscInt *n,PetscInt *p)
{
  PetscInt w;
  DM_DA    *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  /* since the xs, xe ... have all been multiplied by the number of degrees
     of freedom per cell, w = dd->w, we divide that out before returning.*/
  w = dd->w;
  if (x) *x = dd->Xs/w + dd->xo;
  if (y) *y = dd->Ys   + dd->yo;
  if (z) *z = dd->Zs   + dd->zo;
  if (m) *m = (dd->Xe - dd->Xs)/w;
  if (n) *n = (dd->Ye - dd->Ys);
  if (p) *p = (dd->Ze - dd->Zs);
  PetscFunctionReturn(0);
}

