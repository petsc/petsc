#include <petsctao.h> /*I  "petsctao.h"  I*/
#include <petsc/private/taoimpl.h>
#include <petscsnes.h>
#include <petscdmshell.h>

/*
   For finited difference computations of the Hessian, we use PETSc's SNESComputeJacobianDefault
*/
static PetscErrorCode Fsnes(SNES snes, Vec X, Vec G, void *ctx)
{
  Tao tao = (Tao)ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 4);
  PetscCall(TaoComputeGradient(tao, X, G));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoDefaultComputeGradient - computes the gradient using finite differences.

  Collective

  Input Parameters:
+ tao   - the Tao context
. X     - compute gradient at this point
- dummy - not used

  Output Parameter:
. G - Gradient Vector

  Options Database Key:
+ -tao_fd_gradient      - activates TaoDefaultComputeGradient()
- -tao_fd_delta <delta> - change in X used to calculate finite differences

  Level: advanced

  Notes:
  This routine is slow and expensive, and is not optimized
  to take advantage of sparsity in the problem.  Although
  not recommended for general use
  in large-scale applications, it can be useful in checking the
  correctness of a user-provided gradient.  Use the tao method TAOTEST
  to get an indication of whether your gradient is correct.
  This finite difference gradient evaluation can be set using the routine `TaoSetGradient()` or by using the command line option -tao_fd_gradient

.seealso: `Tao`, `TaoSetGradient()`
@*/
PetscErrorCode TaoDefaultComputeGradient(Tao tao, Vec Xin, Vec G, void *dummy)
{
  Vec          X;
  PetscScalar *g;
  PetscReal    f, f2;
  PetscInt     low, high, N, i;
  PetscBool    flg;
  PetscReal    h = .5 * PETSC_SQRT_MACHINE_EPSILON;

  PetscFunctionBegin;
  PetscCall(PetscOptionsGetReal(((PetscObject)tao)->options, ((PetscObject)tao)->prefix, "-tao_fd_delta", &h, &flg));
  PetscCall(VecDuplicate(Xin, &X));
  PetscCall(VecCopy(Xin, X));
  PetscCall(VecGetSize(X, &N));
  PetscCall(VecGetOwnershipRange(X, &low, &high));
  PetscCall(VecSetOption(X, VEC_IGNORE_OFF_PROC_ENTRIES, PETSC_TRUE));
  PetscCall(VecGetArray(G, &g));
  for (i = 0; i < N; i++) {
    PetscCall(VecSetValue(X, i, -h, ADD_VALUES));
    PetscCall(VecAssemblyBegin(X));
    PetscCall(VecAssemblyEnd(X));
    PetscCall(TaoComputeObjective(tao, X, &f));
    PetscCall(VecSetValue(X, i, 2.0 * h, ADD_VALUES));
    PetscCall(VecAssemblyBegin(X));
    PetscCall(VecAssemblyEnd(X));
    PetscCall(TaoComputeObjective(tao, X, &f2));
    PetscCall(VecSetValue(X, i, -h, ADD_VALUES));
    PetscCall(VecAssemblyBegin(X));
    PetscCall(VecAssemblyEnd(X));
    if (i >= low && i < high) g[i - low] = (f2 - f) / (2.0 * h);
  }
  PetscCall(VecRestoreArray(G, &g));
  PetscCall(VecDestroy(&X));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   TaoDefaultComputeHessian - Computes the Hessian using finite differences.

   Collective

   Input Parameters:
+  tao   - the Tao context
.  V     - compute Hessian at this point
-  dummy - not used

   Output Parameters:
+  H - Hessian matrix (not altered in this routine)
-  B - newly computed Hessian matrix to use with preconditioner (generally the same as H)

   Options Database Key:
.  -tao_fd_hessian - activates TaoDefaultComputeHessian()

   Level: advanced

   Notes:
   This routine is slow and expensive, and is not optimized
   to take advantage of sparsity in the problem.  Although
   it is not recommended for general use
   in large-scale applications, It can be useful in checking the
   correctness of a user-provided Hessian.

.seealso: `Tao`, `TaoSetHessian()`, `TaoDefaultComputeHessianColor()`, `SNESComputeJacobianDefault()`, `TaoSetGradient()`, `TaoDefaultComputeGradient()`
@*/
PetscErrorCode TaoDefaultComputeHessian(Tao tao, Vec V, Mat H, Mat B, void *dummy)
{
  SNES snes;
  DM   dm;

  PetscFunctionBegin;
  PetscCall(PetscInfo(tao, "TAO Using finite differences w/o coloring to compute Hessian matrix\n"));
  PetscCall(SNESCreate(PetscObjectComm((PetscObject)H), &snes));
  PetscCall(SNESSetFunction(snes, NULL, Fsnes, tao));
  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(DMShellSetGlobalVector(dm, V));
  PetscCall(SNESSetUp(snes));
  if (H) {
    PetscInt n, N;

    PetscCall(VecGetSize(V, &N));
    PetscCall(VecGetLocalSize(V, &n));
    PetscCall(MatSetSizes(H, n, n, N, N));
    PetscCall(MatSetUp(H));
  }
  if (B && B != H) {
    PetscInt n, N;

    PetscCall(VecGetSize(V, &N));
    PetscCall(VecGetLocalSize(V, &n));
    PetscCall(MatSetSizes(B, n, n, N, N));
    PetscCall(MatSetUp(B));
  }
  PetscCall(SNESComputeJacobianDefault(snes, V, H, B, NULL));
  PetscCall(SNESDestroy(&snes));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   TaoDefaultComputeHessianColor - Computes the Hessian using colored finite differences.

   Collective

   Input Parameters:
+  tao - the Tao context
.  V   - compute Hessian at this point
-  ctx - the color object of type `MatFDColoring`

   Output Parameters:
+  H - Hessian matrix (not altered in this routine)
-  B - newly computed Hessian matrix to use with preconditioner (generally the same as H)

   Level: advanced

.seealso: `Tao`, `MatColoring`, `TaoSetHessian()`, `TaoDefaultComputeHessian()`, `SNESComputeJacobianDefaultColor()`, `TaoSetGradient()`
@*/
PetscErrorCode TaoDefaultComputeHessianColor(Tao tao, Vec V, Mat H, Mat B, void *ctx)
{
  MatFDColoring coloring = (MatFDColoring)ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(coloring, MAT_FDCOLORING_CLASSID, 5);
  PetscCall(PetscInfo(tao, "TAO computing matrix using finite differences Hessian and coloring\n"));
  PetscCall(MatFDColoringApply(B, coloring, V, ctx));
  if (H != B) {
    PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TaoDefaultComputeHessianMFFD(Tao tao, Vec X, Mat H, Mat B, void *ctx)
{
  PetscInt  n, N;
  PetscBool assembled;

  PetscFunctionBegin;
  PetscCheck(!B || B == H, PetscObjectComm((PetscObject)tao), PETSC_ERR_SUP, "Preconditioning Hessian matrix");
  PetscCall(MatAssembled(H, &assembled));
  if (!assembled) {
    PetscCall(VecGetSize(X, &N));
    PetscCall(VecGetLocalSize(X, &n));
    PetscCall(MatSetSizes(H, n, n, N, N));
    PetscCall(MatSetType(H, MATMFFD));
    PetscCall(MatSetUp(H));
    PetscCall(MatMFFDSetFunction(H, (PetscErrorCode(*)(void *, Vec, Vec))TaoComputeGradient, tao));
  }
  PetscCall(MatMFFDSetBase(H, X, NULL));
  PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}
