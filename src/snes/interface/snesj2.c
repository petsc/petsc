
#include <petsc-private/snesimpl.h>    /*I  "petscsnes.h"  I*/
#include <petscdm.h>                   /*I  "petscdm.h"    I*/

#undef __FUNCT__
#define __FUNCT__ "SNESComputeJacobianDefaultColor"
/*@C
    SNESComputeJacobianDefaultColor - Computes the Jacobian using
    finite differences and coloring to exploit matrix sparsity.

    Collective on SNES

    Input Parameters:
+   snes - nonlinear solver object
.   x1 - location at which to evaluate Jacobian
-   ctx - MatFDColoring context or NULL

    Output Parameters:
+   J - Jacobian matrix (not altered in this routine)
.   B - newly computed Jacobian matrix to use with preconditioner (generally the same as J)
-   flag - flag indicating whether the matrix sparsity structure has changed

    Level: intermediate

.notes: If the coloring is not provided through the context, this will first try to get the
        coloring from the DM.  If the DM type has no coloring routine, then it will try to
        get the coloring from the matrix.  This requires that the matrix have nonzero entries
        precomputed.  This is discouraged, as MatGetColoring() is not parallel.

.keywords: SNES, finite differences, Jacobian, coloring, sparse

.seealso: SNESSetJacobian(), SNESTestJacobian(), SNESComputeJacobianDefault()
          MatFDColoringCreate(), MatFDColoringSetFunction()

@*/

PetscErrorCode  SNESComputeJacobianDefaultColor(SNES snes,Vec x1,Mat *J,Mat *B,MatStructure *flag,void *ctx)
{
  MatFDColoring  color = (MatFDColoring)ctx;
  PetscErrorCode ierr;
  DM             dm;
  PetscErrorCode (*func)(SNES,Vec,Vec,void*);
  Vec            F;
  void           *funcctx;
  ISColoring     iscoloring;
  PetscBool      hascolor;
  PetscBool      solvec;

  PetscFunctionBegin;
  if (color) PetscValidHeaderSpecific(color,MAT_FDCOLORING_CLASSID,6);
  else {ierr  = PetscObjectQuery((PetscObject)*B,"SNESMatFDColoring",(PetscObject*)&color);CHKERRQ(ierr);}
  *flag = SAME_NONZERO_PATTERN;
  ierr  = SNESGetFunction(snes,&F,&func,&funcctx);CHKERRQ(ierr);
  if (!color) {
    ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
    ierr = DMHasColoring(dm,&hascolor);CHKERRQ(ierr);
    if (hascolor) {
      ierr = DMCreateColoring(dm,IS_COLORING_GLOBAL,&iscoloring);CHKERRQ(ierr);
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

  /* F is only usable if there is no RHS on the SNES and the full solution corresponds to x1 */
  ierr = VecEqual(x1,snes->vec_sol,&solvec);CHKERRQ(ierr);
  if (!snes->vec_rhs && solvec) {
    ierr = MatFDColoringSetF(color,F);CHKERRQ(ierr);
  }
  ierr = MatFDColoringApply(*B,color,x1,flag,snes);CHKERRQ(ierr);
  if (*J != *B) {
    ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
