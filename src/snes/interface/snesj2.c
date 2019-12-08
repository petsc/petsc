
#include <petsc/private/snesimpl.h>    /*I  "petscsnes.h"  I*/
#include <petscdm.h>                   /*I  "petscdm.h"    I*/

/*
   MatFDColoringSetFunction() takes a function with four arguments, we want to use SNESComputeFunction()
   since it logs function computation information.
*/
static PetscErrorCode SNESComputeFunctionCtx(SNES snes,Vec x,Vec f,void *ctx)
{
  return SNESComputeFunction(snes,x,f);
}

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
-   B - newly computed Jacobian matrix to use with preconditioner (generally the same as J)

    Level: intermediate

   Options Database Key:
+  -snes_fd_color_use_mat - use a matrix coloring from the explicit matrix nonzero pattern instead of from the DM providing the matrix
.  -snes_fd_color - Activates SNESComputeJacobianDefaultColor() in SNESSetFromOptions()
.  -mat_fd_coloring_err <err> - Sets <err> (square root of relative error in the function)
.  -mat_fd_coloring_umin <umin> - Sets umin, the minimum allowable u-value magnitude
.  -mat_fd_type - Either wp or ds (see MATMFFD_WP or MATMFFD_DS)
.  -snes_mf_operator - Use matrix free application of Jacobian
-  -snes_mf - Use matrix free Jacobian with not explicit Jacobian represenation

    Notes:
        If the coloring is not provided through the context, this will first try to get the
        coloring from the DM.  If the DM type has no coloring routine, then it will try to
        get the coloring from the matrix.  This requires that the matrix have nonzero entries
        precomputed.

       SNES supports three approaches for computing (approximate) Jacobians: user provided via SNESSetJacobian(), matrix free via SNESSetUseMatrixFree,
       and computing explictly with finite differences and coloring using MatFDColoring. It is also possible to use automatic differentiation and the MatFDColoring object.


.seealso: SNESSetJacobian(), SNESTestJacobian(), SNESComputeJacobianDefault(), SNESSetUseMatrixFree(),
          MatFDColoringCreate(), MatFDColoringSetFunction()

@*/

PetscErrorCode  SNESComputeJacobianDefaultColor(SNES snes,Vec x1,Mat J,Mat B,void *ctx)
{
  MatFDColoring  color = (MatFDColoring)ctx;
  PetscErrorCode ierr;
  DM             dm;
  MatColoring    mc;
  ISColoring     iscoloring;
  PetscBool      hascolor;
  PetscBool      solvec,matcolor = PETSC_FALSE;

  PetscFunctionBegin;
  if (color) PetscValidHeaderSpecific(color,MAT_FDCOLORING_CLASSID,6);
  if (!color) {ierr  = PetscObjectQuery((PetscObject)B,"SNESMatFDColoring",(PetscObject*)&color);CHKERRQ(ierr);}

  if (!color) {
    ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
    ierr = DMHasColoring(dm,&hascolor);CHKERRQ(ierr);
    matcolor = PETSC_FALSE;
    ierr = PetscOptionsGetBool(((PetscObject)snes)->options,((PetscObject)snes)->prefix,"-snes_fd_color_use_mat",&matcolor,NULL);CHKERRQ(ierr);
    if (hascolor && !matcolor) {
      ierr = DMCreateColoring(dm,IS_COLORING_GLOBAL,&iscoloring);CHKERRQ(ierr);
      ierr = MatFDColoringCreate(B,iscoloring,&color);CHKERRQ(ierr);
      ierr = MatFDColoringSetFunction(color,(PetscErrorCode (*)(void))SNESComputeFunctionCtx,NULL);CHKERRQ(ierr);
      ierr = MatFDColoringSetFromOptions(color);CHKERRQ(ierr);
      ierr = MatFDColoringSetUp(B,iscoloring,color);CHKERRQ(ierr);
      ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);
    } else {
      ierr = MatColoringCreate(B,&mc);CHKERRQ(ierr);
      ierr = MatColoringSetDistance(mc,2);CHKERRQ(ierr);
      ierr = MatColoringSetType(mc,MATCOLORINGSL);CHKERRQ(ierr);
      ierr = MatColoringSetFromOptions(mc);CHKERRQ(ierr);
      ierr = MatColoringApply(mc,&iscoloring);CHKERRQ(ierr);
      ierr = MatColoringDestroy(&mc);CHKERRQ(ierr);
      ierr = MatFDColoringCreate(B,iscoloring,&color);CHKERRQ(ierr);
      ierr = MatFDColoringSetFunction(color,(PetscErrorCode (*)(void))SNESComputeFunctionCtx,NULL);CHKERRQ(ierr);
      ierr = MatFDColoringSetFromOptions(color);CHKERRQ(ierr);
      ierr = MatFDColoringSetUp(B,iscoloring,color);CHKERRQ(ierr);
      ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);
    }
    ierr = PetscObjectCompose((PetscObject)B,"SNESMatFDColoring",(PetscObject)color);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)color);CHKERRQ(ierr);
  }

  /* F is only usable if there is no RHS on the SNES and the full solution corresponds to x1 */
  ierr = VecEqual(x1,snes->vec_sol,&solvec);CHKERRQ(ierr);
  if (!snes->vec_rhs && solvec) {
    Vec F;
    ierr = SNESGetFunction(snes,&F,NULL,NULL);CHKERRQ(ierr);
    ierr = MatFDColoringSetF(color,F);CHKERRQ(ierr);
  }
  ierr = MatFDColoringApply(B,color,x1,snes);CHKERRQ(ierr);
  if (J != B) {
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
