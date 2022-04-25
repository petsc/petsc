
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
static PetscErrorCode SNESComputeMFFunctionCtx(SNES snes,Vec x,Vec f,void *ctx)
{
  return SNESComputeMFFunction(snes,x,f);
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
-  -snes_mf - Use matrix free Jacobian with no explicit Jacobian representation

    Notes:
        If the coloring is not provided through the context, this will first try to get the
        coloring from the DM.  If the DM type has no coloring routine, then it will try to
        get the coloring from the matrix.  This requires that the matrix have nonzero entries
        precomputed.

       SNES supports three approaches for computing (approximate) Jacobians: user provided via SNESSetJacobian(), matrix free via SNESSetUseMatrixFree(),
       and computing explicitly with finite differences and coloring using MatFDColoring. It is also possible to use automatic differentiation
       and the MatFDColoring object, see src/ts/tutorials/autodiff/ex16adj_tl.cxx

.seealso: `SNESSetJacobian()`, `SNESTestJacobian()`, `SNESComputeJacobianDefault()`, `SNESSetUseMatrixFree()`,
          `MatFDColoringCreate()`, `MatFDColoringSetFunction()`

@*/
PetscErrorCode  SNESComputeJacobianDefaultColor(SNES snes,Vec x1,Mat J,Mat B,void *ctx)
{
  MatFDColoring  color = (MatFDColoring)ctx;
  DM             dm;
  MatColoring    mc;
  ISColoring     iscoloring;
  PetscBool      hascolor;
  PetscBool      solvec,matcolor = PETSC_FALSE;
  DMSNES         dms;

  PetscFunctionBegin;
  if (color) PetscValidHeaderSpecific(color,MAT_FDCOLORING_CLASSID,5);
  if (!color) PetscCall(PetscObjectQuery((PetscObject)B,"SNESMatFDColoring",(PetscObject*)&color));

  if (!color) {
    PetscCall(SNESGetDM(snes,&dm));
    PetscCall(DMHasColoring(dm,&hascolor));
    matcolor = PETSC_FALSE;
    PetscCall(PetscOptionsGetBool(((PetscObject)snes)->options,((PetscObject)snes)->prefix,"-snes_fd_color_use_mat",&matcolor,NULL));
    if (hascolor && !matcolor) {
      PetscCall(DMCreateColoring(dm,IS_COLORING_GLOBAL,&iscoloring));
    } else {
      PetscCall(MatColoringCreate(B,&mc));
      PetscCall(MatColoringSetDistance(mc,2));
      PetscCall(MatColoringSetType(mc,MATCOLORINGSL));
      PetscCall(MatColoringSetFromOptions(mc));
      PetscCall(MatColoringApply(mc,&iscoloring));
      PetscCall(MatColoringDestroy(&mc));
    }
    PetscCall(MatFDColoringCreate(B,iscoloring,&color));
    PetscCall(DMGetDMSNES(dm,&dms));
    if (dms->ops->computemffunction) {
      PetscCall(MatFDColoringSetFunction(color,(PetscErrorCode (*)(void))SNESComputeMFFunctionCtx,NULL));
    } else {
      PetscCall(MatFDColoringSetFunction(color,(PetscErrorCode (*)(void))SNESComputeFunctionCtx,NULL));
    }
    PetscCall(MatFDColoringSetFromOptions(color));
    PetscCall(MatFDColoringSetUp(B,iscoloring,color));
    PetscCall(ISColoringDestroy(&iscoloring));
    PetscCall(PetscObjectCompose((PetscObject)B,"SNESMatFDColoring",(PetscObject)color));
    PetscCall(PetscObjectDereference((PetscObject)color));
  }

  /* F is only usable if there is no RHS on the SNES and the full solution corresponds to x1 */
  PetscCall(VecEqual(x1,snes->vec_sol,&solvec));
  if (!snes->vec_rhs && solvec) {
    Vec F;
    PetscCall(SNESGetFunction(snes,&F,NULL,NULL));
    PetscCall(MatFDColoringSetF(color,F));
  }
  PetscCall(MatFDColoringApply(B,color,x1,snes));
  if (J != B) {
    PetscCall(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}
