/*$Id: cmesh.c,v 1.70 2000/05/05 22:14:53 balay Exp bsmith $*/

#include "petscvec.h"        /*I "petscvec.h" I*/


#undef __FUNC__  
#define __FUNC__ "VecContourScale"
/*@
    VecContourScale - Prepares a vector of values to be plotted using 
    the PetscDrawTriangle() contour plotter.

    Collective on Vec

    Input Parameters:
+   v - the vector of values
.   vmin - minimum value (for lowest color)
-   vmax - maximum value (for highest color)

   Level: intermediate

.seealso: PetscDrawTensorContour(),PetscDrawTensorContourPatch()

@*/
int VecContourScale(Vec v,PetscReal vmin,PetscReal vmax)
{
  Scalar *values;
  int    ierr,n,i;
  PetscReal scale;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_COOKIE);

  if (PetscAbsDouble(vmax - vmin) < 1.e-50) {
     scale = 1.0;
  } else {
    scale = (245.0 - PETSC_DRAW_BASIC_COLORS)/(vmax - vmin); 
  }

  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  ierr = VecGetArray(v,&values);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    values[i] = (PetscReal)PETSC_DRAW_BASIC_COLORS + scale*(values[i] - vmin);
  }
  ierr = VecRestoreArray(v,&values);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
