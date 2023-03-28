
#include <petsc/private/drawimpl.h> /*I   "petscdraw.h"  I*/

/*@
   PetscDrawLGAddCommonPoint - Adds another point to each of the line graphs. All the points share
      the same new X coordinate.  The new point must have an X coordinate larger than the old points.

   Logically Collective

   Input Parameters:
+  lg - the line graph context
.   x - the common x coordinate point
-   y - the new y coordinate point for each curve.

   Level: intermediate

   Notes:
   You must call `PetscDrawLGDraw()` to display any added points

   Call `PetscDrawLGReset()` to remove all points

.seealso: `PetscDrawLG`, `PetscDrawLGCreate()`, `PetscDrawLGAddPoints()`, `PetscDrawLGAddPoint()`, `PetscDrawLGReset()`, `PetscDrawLGDraw()`
@*/
PetscErrorCode PetscDrawLGAddCommonPoint(PetscDrawLG lg, const PetscReal x, const PetscReal *y)
{
  PetscInt i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lg, PETSC_DRAWLG_CLASSID, 1);

  if (lg->loc + lg->dim >= lg->len) { /* allocate more space */
    PetscReal *tmpx, *tmpy;
    PetscCall(PetscMalloc2(lg->len + lg->dim * PETSC_DRAW_LG_CHUNK_SIZE, &tmpx, lg->len + lg->dim * PETSC_DRAW_LG_CHUNK_SIZE, &tmpy));
    PetscCall(PetscArraycpy(tmpx, lg->x, lg->len));
    PetscCall(PetscArraycpy(tmpy, lg->y, lg->len));
    PetscCall(PetscFree2(lg->x, lg->y));
    lg->x = tmpx;
    lg->y = tmpy;
    lg->len += lg->dim * PETSC_DRAW_LG_CHUNK_SIZE;
  }
  for (i = 0; i < lg->dim; i++) {
    if (x > lg->xmax) lg->xmax = x;
    if (x < lg->xmin) lg->xmin = x;
    if (y[i] > lg->ymax) lg->ymax = y[i];
    if (y[i] < lg->ymin) lg->ymin = y[i];

    lg->x[lg->loc]   = x;
    lg->y[lg->loc++] = y[i];
  }
  lg->nopts++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PetscDrawLGAddPoint - Adds another point to each of the line graphs.
   The new point must have an X coordinate larger than the old points.

   Logically Collective

   Input Parameters:
+  lg - the line graph context
.  x - array containing the x coordinate for the point on each curve
-  y - array containing the y coordinate for the point on each curve

   Level: intermediate

   Notes:
   You must call `PetscDrawLGDraw()` to display any added points

   Call `PetscDrawLGReset()` to remove all points

.seealso: `PetscDrawLG`, `PetscDrawLGCreate()`, `PetscDrawLGAddPoints()`, `PetscDrawLGAddCommonPoint()`, `PetscDrawLGReset()`, `PetscDrawLGDraw()`
@*/
PetscErrorCode PetscDrawLGAddPoint(PetscDrawLG lg, const PetscReal *x, const PetscReal *y)
{
  PetscInt  i;
  PetscReal xx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lg, PETSC_DRAWLG_CLASSID, 1);

  if (lg->loc + lg->dim >= lg->len) { /* allocate more space */
    PetscReal *tmpx, *tmpy;
    PetscCall(PetscMalloc2(lg->len + lg->dim * PETSC_DRAW_LG_CHUNK_SIZE, &tmpx, lg->len + lg->dim * PETSC_DRAW_LG_CHUNK_SIZE, &tmpy));
    PetscCall(PetscArraycpy(tmpx, lg->x, lg->len));
    PetscCall(PetscArraycpy(tmpy, lg->y, lg->len));
    PetscCall(PetscFree2(lg->x, lg->y));
    lg->x = tmpx;
    lg->y = tmpy;
    lg->len += lg->dim * PETSC_DRAW_LG_CHUNK_SIZE;
  }
  for (i = 0; i < lg->dim; i++) {
    if (!x) {
      xx = lg->nopts;
    } else {
      xx = x[i];
    }
    if (xx > lg->xmax) lg->xmax = xx;
    if (xx < lg->xmin) lg->xmin = xx;
    if (y[i] > lg->ymax) lg->ymax = y[i];
    if (y[i] < lg->ymin) lg->ymin = y[i];

    lg->x[lg->loc]   = xx;
    lg->y[lg->loc++] = y[i];
  }
  lg->nopts++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscDrawLGAddPoints - Adds several points to each of the line graphs.
   The new points must have an X coordinate larger than the old points.

   Logically Collective

   Input Parameters:
+  lg - the line graph context
.  xx - array of pointers that point to arrays containing the new x coordinates for each curve.
.  yy - array of pointers that point to arrays containing the new y points for each curve.
-  n - number of points being added

   Level: intermediate

   Notes:
   You must call `PetscDrawLGDraw()` to display any added points

   Call `PetscDrawLGReset()` to remove all points

.seealso: `PetscDrawLG`, `PetscDrawLGCreate()`, `PetscDrawLGAddPoint()`, `PetscDrawLGAddCommonPoint()`, `PetscDrawLGReset()`, `PetscDrawLGDraw()`
@*/
PetscErrorCode PetscDrawLGAddPoints(PetscDrawLG lg, PetscInt n, PetscReal **xx, PetscReal **yy)
{
  PetscInt   i, j, k;
  PetscReal *x, *y;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lg, PETSC_DRAWLG_CLASSID, 1);

  if (lg->loc + n * lg->dim >= lg->len) { /* allocate more space */
    PetscReal *tmpx, *tmpy;
    PetscInt   chunk = PETSC_DRAW_LG_CHUNK_SIZE;

    if (n > chunk) chunk = n;
    PetscCall(PetscMalloc2(lg->len + lg->dim * chunk, &tmpx, lg->len + lg->dim * chunk, &tmpy));
    PetscCall(PetscArraycpy(tmpx, lg->x, lg->len));
    PetscCall(PetscArraycpy(tmpy, lg->y, lg->len));
    PetscCall(PetscFree2(lg->x, lg->y));
    lg->x = tmpx;
    lg->y = tmpy;
    lg->len += lg->dim * chunk;
  }
  for (j = 0; j < lg->dim; j++) {
    x = xx[j];
    y = yy[j];
    k = lg->loc + j;
    for (i = 0; i < n; i++) {
      if (x[i] > lg->xmax) lg->xmax = x[i];
      if (x[i] < lg->xmin) lg->xmin = x[i];
      if (y[i] > lg->ymax) lg->ymax = y[i];
      if (y[i] < lg->ymin) lg->ymin = y[i];

      lg->x[k] = x[i];
      lg->y[k] = y[i];
      k += lg->dim;
    }
  }
  lg->loc += n * lg->dim;
  lg->nopts += n;
  PetscFunctionReturn(PETSC_SUCCESS);
}
