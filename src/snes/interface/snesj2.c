
#include <petsc-private/snesimpl.h>    /*I  "petscsnes.h"  I*/

#undef __FUNCT__
#define __FUNCT__ "SNESDefaultComputeJacobianColor"
/*@C
    SNESDefaultComputeJacobianColor - Computes the Jacobian using
    finite differences and coloring to exploit matrix sparsity.

    Collective on SNES

    Input Parameters:
+   snes - nonlinear solver object
.   x1 - location at which to evaluate Jacobian
-   ctx - coloring context, where ctx must have type MatFDColoring,
          as created via MatFDColoringCreate()

    Output Parameters:
+   J - Jacobian matrix (not altered in this routine)
.   B - newly computed Jacobian matrix to use with preconditioner (generally the same as J)
-   flag - flag indicating whether the matrix sparsity structure has changed

    Level: intermediate

.notes: If ctx is not provided, this will try to get the coloring from the DM.  If the DM type
        has no coloring routine, then it will try to get the coloring from the matrix.  This
        requires that the matrix have nonzero entries precomputed, such as in
        snes/examples/tutorials/ex45.c.

.keywords: SNES, finite differences, Jacobian, coloring, sparse

.seealso: SNESSetJacobian(), SNESTestJacobian(), SNESDefaultComputeJacobian()
          MatFDColoringCreate(), MatFDColoringSetFunction()

@*/

PetscErrorCode  SNESDefaultComputeJacobianColor(SNES snes,Vec x1,Mat *J,Mat *B,MatStructure *flag,void *ctx)
{
  MatFDColoring  color = (MatFDColoring)ctx;
  PetscErrorCode ierr;
  DM dm;
  PetscErrorCode              (*func)(SNES,Vec,Vec,void*);
  Vec                         F;
  void                        *funcctx;
  ISColoring                  iscoloring;
  PetscBool                   hascolor;

  PetscFunctionBegin;
  if (color) {
    PetscValidHeaderSpecific(color,MAT_FDCOLORING_CLASSID,6);
  } else {
    ierr = PetscObjectQuery((PetscObject)*B,"SNESMatFDColoring",(PetscObject *)&color);CHKERRQ(ierr);
  }
  *flag = SAME_NONZERO_PATTERN;
  ierr = SNESGetFunction(snes,&F,&func,&funcctx);
  if (!color) {
    ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
    ierr = DMHasColoring(dm,&hascolor);CHKERRQ(ierr);
    if (hascolor) {
      ierr = DMCreateColoring(dm,IS_COLORING_GLOBAL,MATAIJ,&iscoloring);CHKERRQ(ierr);
      ierr = MatFDColoringCreate(*B,iscoloring,&color);CHKERRQ(ierr);
      ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);
      ierr = MatFDColoringSetFunction(color,(PetscErrorCode (*)(void))func,funcctx);CHKERRQ(ierr);
      ierr = MatFDColoringSetFromOptions(color);CHKERRQ(ierr);
    } else {
      ierr = MatGetColoring(*B,MATCOLORINGSL,&iscoloring);CHKERRQ(ierr);
      ierr = MatFDColoringCreate(*B,iscoloring,&color);CHKERRQ(ierr);
      ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);
      ierr = MatFDColoringSetFunction(color,(PetscErrorCode (*)(void))func,(void*)funcctx);CHKERRQ(ierr);
      ierr = MatFDColoringSetFromOptions(color);CHKERRQ(ierr);
    }
    ierr = PetscObjectCompose((PetscObject)*B,"SNESMatFDColoring",(PetscObject)color);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)color);CHKERRQ(ierr);
  }
  ierr  = MatFDColoringSetF(color,PETSC_NULL);CHKERRQ(ierr);
  ierr  = MatFDColoringApply(*B,color,x1,flag,snes);CHKERRQ(ierr);
  if (*J != *B) {
    ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
