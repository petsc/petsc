#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: cmesh.c,v 1.63 1999/03/02 16:37:52 bsmith Exp bsmith $";
#endif

#include "vec.h"        /*I "vec.h" I*/


#undef __FUNC__  
#define __FUNC__ "VecContourScale"
/*@
    VecContourScale - Prepares a vector of values to be plotted using 
    the DrawTriangle() contour plotter.

    Collective on Vec

    Input Parameters:
+   v - the vector of values
.   vmin - minimum value (for lowest color)
-   vmax - maximum value (for highest color)

@*/
int VecContourScale(Vec v,double vmin,double vmax)
{
  Scalar *values;
  int    ierr,n,i;
  double scale;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_COOKIE);

  if (PetscAbsDouble(vmax - vmin) < 1.e-50) {
     scale = 1.0;
  } else {
    scale = (245.0 - DRAW_BASIC_COLORS)/(vmax - vmin); 
  }

  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  ierr = VecGetArray(v,&values);CHKERRQ(ierr);
  for ( i=0; i<n; i++ ) {
    values[i] = (double)DRAW_BASIC_COLORS + scale*(values[i] - vmin);
  }
  ierr = VecRestoreArray(v,&values);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
