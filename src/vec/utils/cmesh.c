/*$Id: cmesh.c,v 1.68 2000/04/09 04:35:20 bsmith Exp bsmith $*/

#include "vec.h"        /*I "vec.h" I*/


#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"VecContourScale"
/*@
    VecContourScale - Prepares a vector of values to be plotted using 
    the DrawTriangle() contour plotter.

    Collective on Vec

    Input Parameters:
+   v - the vector of values
.   vmin - minimum value (for lowest color)
-   vmax - maximum value (for highest color)

   Level: intermediate

.seealso: DrawTensorContour(),DrawTensorContourPatch()

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
    scale = (245.0 - DRAW_BASIC_COLORS)/(vmax - vmin); 
  }

  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  ierr = VecGetArray(v,&values);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    values[i] = (PetscReal)DRAW_BASIC_COLORS + scale*(values[i] - vmin);
  }
  ierr = VecRestoreArray(v,&values);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
