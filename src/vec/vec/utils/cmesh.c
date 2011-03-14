
#include <petscvec.h>        /*I "petscvec.h" I*/


#undef __FUNCT__  
#define __FUNCT__ "VecContourScale"
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
PetscErrorCode  VecContourScale(Vec v,PetscReal vmin,PetscReal vmax)
{
  PetscScalar    *values;
  PetscErrorCode ierr;
  PetscInt       n,i;
  PetscReal      scale;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);

  if (PetscAbsReal(vmax - vmin) < 1.e-50) {
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
