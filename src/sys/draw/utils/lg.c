
#include <../src/sys/draw/utils/lgimpl.h>

#undef __FUNCT__
#define __FUNCT__ "PetscDrawLGAddCommonPoint"
/*@
   PetscDrawLGAddCommonPoint - Adds another point to each of the line graphs. All the points share
      the same new X coordinate.  The new point must have an X coordinate larger than the old points.

   Not Collective, but ignored by all processors except processor 0 in PetscDrawLG

   Input Parameters:
+  lg - the LineGraph data structure
.   x - the common x coordiante point
-   y - the new y coordinate point for each curve.

   Level: intermediate

   Concepts: line graph^adding points

.seealso: PetscDrawLGAddPoints(), PetscDrawLGAddPoint()
@*/
PetscErrorCode  PetscDrawLGAddCommonPoint(PetscDrawLG lg,PetscReal x,const PetscReal *y)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (lg && ((PetscObject)lg)->classid == PETSC_DRAW_CLASSID) PetscFunctionReturn(0);

  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,1);
  if (lg->loc+lg->dim >= lg->len) { /* allocate more space */
    PetscReal *tmpx,*tmpy;
    ierr = PetscMalloc2(lg->len+lg->dim*CHUNCKSIZE,PetscReal,&tmpx,lg->len+lg->dim*CHUNCKSIZE,PetscReal,&tmpy);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(lg,2*lg->dim*CHUNCKSIZE*sizeof(PetscReal));CHKERRQ(ierr);
    ierr = PetscMemcpy(tmpx,lg->x,lg->len*sizeof(PetscReal));CHKERRQ(ierr);
    ierr = PetscMemcpy(tmpy,lg->y,lg->len*sizeof(PetscReal));CHKERRQ(ierr);
    ierr = PetscFree2(lg->x,lg->y);CHKERRQ(ierr);
    lg->x = tmpx;
    lg->y = tmpy;
    lg->len += lg->dim*CHUNCKSIZE;
  }
  for (i=0; i<lg->dim; i++) {
    if (x > lg->xmax) lg->xmax = x;
    if (x < lg->xmin) lg->xmin = x;
    if (y[i] > lg->ymax) lg->ymax = y[i];
    if (y[i] < lg->ymin) lg->ymin = y[i];

    lg->x[lg->loc]   = x;
    lg->y[lg->loc++] = y[i];
  }
  lg->nopts++;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawLGAddPoint"
/*@
   PetscDrawLGAddPoint - Adds another point to each of the line graphs.
   The new point must have an X coordinate larger than the old points.

   Not Collective, but ignored by all processors except processor 0 in PetscDrawLG

   Input Parameters:
+  lg - the LineGraph data structure
-  x, y - the points to two vectors containing the new x and y
          point for each curve.

   Level: intermediate

   Concepts: line graph^adding points

.seealso: PetscDrawLGAddPoints(), PetscDrawLGAddCommonPoint()
@*/
PetscErrorCode  PetscDrawLGAddPoint(PetscDrawLG lg,const PetscReal *x,const PetscReal *y)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (lg && ((PetscObject)lg)->classid == PETSC_DRAW_CLASSID) PetscFunctionReturn(0);

  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,1);
  if (lg->loc+lg->dim >= lg->len) { /* allocate more space */
    PetscReal *tmpx,*tmpy;
    ierr = PetscMalloc2(lg->len+lg->dim*CHUNCKSIZE,PetscReal,&tmpx,lg->len+lg->dim*CHUNCKSIZE,PetscReal,&tmpy);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(lg,2*lg->dim*CHUNCKSIZE*sizeof(PetscReal));CHKERRQ(ierr);
    ierr = PetscMemcpy(tmpx,lg->x,lg->len*sizeof(PetscReal));CHKERRQ(ierr);
    ierr = PetscMemcpy(tmpy,lg->y,lg->len*sizeof(PetscReal));CHKERRQ(ierr);
    ierr = PetscFree2(lg->x,lg->y);CHKERRQ(ierr);
    lg->x = tmpx;
    lg->y = tmpy;
    lg->len += lg->dim*CHUNCKSIZE;
  }
  for (i=0; i<lg->dim; i++) {
    if (x[i] > lg->xmax) lg->xmax = x[i];
    if (x[i] < lg->xmin) lg->xmin = x[i];
    if (y[i] > lg->ymax) lg->ymax = y[i];
    if (y[i] < lg->ymin) lg->ymin = y[i];

    lg->x[lg->loc]   = x[i];
    lg->y[lg->loc++] = y[i];
  }
  lg->nopts++;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawLGAddPoints"
/*@C
   PetscDrawLGAddPoints - Adds several points to each of the line graphs.
   The new points must have an X coordinate larger than the old points.

   Not Collective, but ignored by all processors except processor 0 in PetscDrawLG

   Input Parameters:
+  lg - the LineGraph data structure
.  xx,yy - points to two arrays of pointers that point to arrays
           containing the new x and y points for each curve.
-  n - number of points being added

   Level: intermediate


   Concepts: line graph^adding points

.seealso: PetscDrawLGAddPoint()
@*/
PetscErrorCode  PetscDrawLGAddPoints(PetscDrawLG lg,PetscInt n,PetscReal **xx,PetscReal **yy)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k;
  PetscReal      *x,*y;

  PetscFunctionBegin;
  if (lg && ((PetscObject)lg)->classid == PETSC_DRAW_CLASSID) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,1);
  if (lg->loc+n*lg->dim >= lg->len) { /* allocate more space */
    PetscReal *tmpx,*tmpy;
    PetscInt  chunk = CHUNCKSIZE;

    if (n > chunk) chunk = n;
    ierr = PetscMalloc2(lg->len+lg->dim*chunk,PetscReal,&tmpx,lg->len+lg->dim*chunk,PetscReal,&tmpy);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(lg,2*lg->dim*chunk*sizeof(PetscReal));CHKERRQ(ierr);
    ierr = PetscMemcpy(tmpx,lg->x,lg->len*sizeof(PetscReal));CHKERRQ(ierr);
    ierr = PetscMemcpy(tmpy,lg->y,lg->len*sizeof(PetscReal));CHKERRQ(ierr);
    ierr = PetscFree2(lg->x,lg->y);CHKERRQ(ierr);
    lg->x = tmpx;
    lg->y = tmpy;
    lg->len += lg->dim*chunk;
  }
  for (j=0; j<lg->dim; j++) {
    x = xx[j]; y = yy[j];
    k = lg->loc + j;
    for (i=0; i<n; i++) {
      if (x[i] > lg->xmax) lg->xmax = x[i];
      if (x[i] < lg->xmin) lg->xmin = x[i];
      if (y[i] > lg->ymax) lg->ymax = y[i];
      if (y[i] < lg->ymin) lg->ymin = y[i];

      lg->x[k]   = x[i];
      lg->y[k] = y[i];
      k += lg->dim;
    }
  }
  lg->loc   += n*lg->dim;
  lg->nopts += n;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawLGSetLimits"
/*@
   PetscDrawLGSetLimits - Sets the axis limits for a line graph. If more
   points are added after this call, the limits will be adjusted to
   include those additional points.

   Not Collective, but ignored by all processors except processor 0 in PetscDrawLG

   Input Parameters:
+  xlg - the line graph context
-  x_min,x_max,y_min,y_max - the limits

   Level: intermediate

   Concepts: line graph^setting axis

@*/
PetscErrorCode  PetscDrawLGSetLimits(PetscDrawLG lg,PetscReal x_min,PetscReal x_max,PetscReal y_min,PetscReal y_max)
{
  PetscFunctionBegin;
  if (lg && ((PetscObject)lg)->classid == PETSC_DRAW_CLASSID) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,1);
  (lg)->xmin = x_min;
  (lg)->xmax = x_max;
  (lg)->ymin = y_min;
  (lg)->ymax = y_max;
  PetscFunctionReturn(0);
}

