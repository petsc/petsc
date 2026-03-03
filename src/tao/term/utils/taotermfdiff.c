#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/
#include <petscsnes.h>
#include <petscdmshell.h>

typedef struct _n_TaoTermWithParameters {
  TaoTerm term; // weak-reference
  Vec     params;
} TaoTermWithParameters;

PETSC_INTERN PetscErrorCode TaoTermWithParametersDestroy(PetscCtxRt ctx)
{
  TaoTermWithParameters *t = (TaoTermWithParameters *)*(void **)ctx;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&t->params));
  PetscCall(PetscFree(t));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESFunction_TaoTerm(SNES snes, Vec X, Vec G, void *ctx)
{
  TaoTermWithParameters *t = (TaoTermWithParameters *)ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(t->term, TAOTERM_CLASSID, 4);
  PetscCall(TaoTermComputeGradient(t->term, X, t->params, G));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermComputeGradientFD - Approximate the gradient of a `TaoTerm` using finite differences

  Collective

  Input Parameters:
+ term   - a `TaoTerm`
. x      - a solution vector
- params - parameters vector (may be `NULL`, see `TaoTermParametersMode`)

  Output Parameter:
. g - the computed finite difference approximation to the gradient

  Options Database Keys:
+ -tao_term_fd_delta <delta>       - change in `x` used to calculate finite differences
- -tao_term_gradient_use_fd <bool> - Use `TaoTermComputeGradientFD()` in `TaoTermComputeGradient()`

  Level: advanced

  Notes:
  This routine is slow and expensive, and is not optimized to take advantage of
  sparsity in the problem.  Although not recommended for general use in
  large-scale applications, it can be useful in checking the correctness of a
  user-provided gradient.  Call `TaoTermComputeGradientSetUseFD()` to start using
  this routine in `TaoTermComputeGradient()`.

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermGetFDDelta()`,
          `TaoTermSetFDDelta()`,
          `TaoTermComputeGradientSetUseFD()`,
          `TaoTermComputeGradientGetUseFD()`,
          `TaoTermComputeHessianFD()`
@*/
PetscErrorCode TaoTermComputeGradientFD(TaoTerm term, Vec x, Vec params, Vec g)
{
  Vec          x_perturbed;
  PetscScalar *_g;
  PetscReal    f, f2;
  PetscInt     low, high, N, i;
  PetscReal    h;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  if (params) PetscValidHeaderSpecific(params, VEC_CLASSID, 3);
  PetscValidHeaderSpecific(g, VEC_CLASSID, 4);
  h = term->fd_delta;

  PetscCall(VecDuplicate(x, &x_perturbed));
  PetscCall(VecCopy(x, x_perturbed));
  PetscCall(VecGetSize(x_perturbed, &N));
  PetscCall(VecGetOwnershipRange(x_perturbed, &low, &high));
  PetscCall(VecSetOption(x_perturbed, VEC_IGNORE_OFF_PROC_ENTRIES, PETSC_TRUE));
  PetscCall(VecGetArray(g, &_g));
  for (i = 0; i < N; i++) {
    PetscCall(VecSetValue(x_perturbed, i, -h, ADD_VALUES));
    PetscCall(VecAssemblyBegin(x_perturbed));
    PetscCall(VecAssemblyEnd(x_perturbed));
    PetscCall(TaoTermComputeObjective(term, x_perturbed, params, &f));
    PetscCall(VecSetValue(x_perturbed, i, 2.0 * h, ADD_VALUES));
    PetscCall(VecAssemblyBegin(x_perturbed));
    PetscCall(VecAssemblyEnd(x_perturbed));
    PetscCall(TaoTermComputeObjective(term, x_perturbed, params, &f2));
    PetscCall(VecSetValue(x_perturbed, i, -h, ADD_VALUES));
    PetscCall(VecAssemblyBegin(x_perturbed));
    PetscCall(VecAssemblyEnd(x_perturbed));
    if (i >= low && i < high) _g[i - low] = (f2 - f) / (2.0 * h);
  }
  PetscCall(VecRestoreArray(g, &_g));
  PetscCall(VecDestroy(&x_perturbed));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermComputeHessianFD - Use finite difference to compute Hessian matrix.

  Collective

  Input Parameters:
+ term   - a `TaoTerm`
. x      - a solution vector
- params - parameters vector (may be `NULL`, see `TaoTermParametersMode`)

  Output Parameters:
+ H    - (optional) Hessian matrix
- Hpre - (optional) Hessian preconditioning matrix

  Options Database Keys:
+ -tao_term_fd_delta <delta>      - change in X used to calculate finite differences
- -tao_term_hessian_use_fd <bool> - Use `TaoTermComputeHessianFD()` in `TaoTermComputeHessian()`

  Level: advanced

  Notes:
  This routine is slow and expensive, and is not optimized to take advantage of
  sparsity in the problem.  Although not recommended for general use in
  large-scale applications, it can be useful in checking the correctness of a
  user-provided Hessian.  Call `TaoTermComputeHessianSetUseFD()` to start using
  this routine in `TaoTermComputeHessian()`.

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermComputeHessian()`,
          `TaoTermGetFDDelta()`,
          `TaoTermSetFDDelta()`,
          `TaoTermComputeHessianSetUseFD()`,
          `TaoTermComputeHessianGetUseFD()`
@*/
PetscErrorCode TaoTermComputeHessianFD(TaoTerm term, Vec x, Vec params, Mat H, Mat Hpre)
{
  SNES                  snes;
  DM                    dm;
  TaoTermWithParameters t;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  if (params) PetscValidHeaderSpecific(params, VEC_CLASSID, 3);
  if (H) PetscValidHeaderSpecific(H, MAT_CLASSID, 4);
  if (Hpre) PetscValidHeaderSpecific(Hpre, MAT_CLASSID, 5);
  PetscCheck(H || Hpre, PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_NULL, "At least one of H or Hpre must be non-NULL");
  // Note: Request for FD takes higher precedence over MATMFFD. Ignore MFFD
  PetscCall(PetscInfo(term, "%s: TaoTerm using finite differences w/o coloring to compute Hessian matrix.\n", ((PetscObject)term)->prefix));
  // Note: same routine as in fdiff.c
  PetscCall(SNESCreate(PetscObjectComm((PetscObject)term), &snes));

  t.term   = term;
  t.params = params;
  PetscCall(SNESSetFunction(snes, NULL, SNESFunction_TaoTerm, (void *)&t));
  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(DMShellSetGlobalVector(dm, x));
  PetscCall(SNESSetUp(snes));
  if (H) {
    PetscInt n, N;

    PetscCall(VecGetSize(x, &N));
    PetscCall(VecGetLocalSize(x, &n));
    PetscCall(MatSetSizes(H, n, n, N, N));
    PetscCall(MatSetUp(H));
  }
  if (Hpre && Hpre != H) {
    PetscInt n, N;

    PetscCall(VecGetSize(x, &N));
    PetscCall(VecGetLocalSize(x, &n));
    PetscCall(MatSetSizes(Hpre, n, n, N, N));
    PetscCall(MatSetUp(Hpre));
  }
  PetscCall(SNESComputeJacobianDefault(snes, x, H ? H : Hpre, Hpre ? Hpre : H, NULL));
  PetscCall(SNESDestroy(&snes));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMFFDFunction_TaoTermHessianShell(void *ctx, Vec x, Vec g)
{
  TaoTermWithParameters *tp = (TaoTermWithParameters *)ctx;

  PetscFunctionBegin;
  // we expect the solution to move around in a finite difference method, but not the parameters
  // TODO but not checking for it now
  PetscCall(TaoTermComputeGradient(tp->term, x, tp->params, g));
  tp->term->ngrad_mffd++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermInitializeHessianMFFD(TaoTerm term, Mat mffd)
{
  TaoTermWithParameters *tp;
  PetscLayout            sol_layout;
  VecType                sol_vec_type;
  PetscContainer         container;

  PetscFunctionBegin;
  PetscCall(TaoTermGetSolutionLayout(term, &sol_layout));
  PetscCall(MatSetLayouts(mffd, sol_layout, sol_layout));
  PetscCall(TaoTermGetSolutionVecType(term, &sol_vec_type));
  PetscCall(MatSetVecType(mffd, sol_vec_type));
  PetscCall(MatSetType(mffd, MATMFFD));
  PetscCall(PetscNew(&tp));
  tp->term = term;
  PetscCall(MatMFFDSetFunction(mffd, MatMFFDFunction_TaoTermHessianShell, tp));
  PetscCall(MatSetOption(mffd, MAT_SYMMETRIC, PETSC_TRUE));
  PetscCall(MatSetOption(mffd, MAT_SYMMETRY_ETERNAL, PETSC_TRUE));
  PetscCall(PetscContainerCreate(PetscObjectComm((PetscObject)term), &container));
  PetscCall(PetscContainerSetPointer(container, (void *)tp));
  PetscCall(PetscContainerSetCtxDestroy(container, TaoTermWithParametersDestroy));
  PetscCall(PetscObjectCompose((PetscObject)mffd, "__TaoTermWithParameters", (PetscObject)container));
  PetscCall(PetscContainerDestroy(&container));
  PetscCall(PetscFree(term->Hpre_mattype));
  PetscCall(PetscFree(term->H_mattype));
  PetscCall(PetscStrallocpy(MATMFFD, (char **)&term->H_mattype));
  PetscCall(PetscStrallocpy(MATMFFD, (char **)&term->Hpre_mattype));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermCreateHessianMFFD - Create a `MATMFFD` for a matrix-free finite-difference approximation of the Hessian of a `TaoTerm`

  Collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameter:
. mffd - a `Mat` of type `MATMFFD`

  Level: advanced

.seealso: [](sec_tao_term), `TaoTerm`, `TaoTermComputeHessianFD()`
@*/
PetscErrorCode TaoTermCreateHessianMFFD(TaoTerm term, Mat *mffd)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(mffd, 2);
  PetscCall(MatCreate(PetscObjectComm((PetscObject)term), mffd));
  PetscCall(TaoTermInitializeHessianMFFD(term, *mffd));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TaoTermComputeHessianMFFD(TaoTerm term, Vec x, Vec params, Mat H, Mat B)
{
  PetscContainer         container;
  TaoTermWithParameters *tp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  if (params) PetscValidHeaderSpecific(params, VEC_CLASSID, 3);
  PetscValidHeaderSpecific(H, MAT_CLASSID, 4);
  if (B) PetscValidHeaderSpecific(B, MAT_CLASSID, 5);
  PetscCall(PetscObjectQuery((PetscObject)H, "__TaoTermWithParameters", (PetscObject *)&container));
  if (!container) {
    PetscCall(TaoTermInitializeHessianMFFD(term, H));
    PetscCall(PetscObjectQuery((PetscObject)H, "__TaoTermWithParameters", (PetscObject *)&container));
    PetscCheck(container, PetscObjectComm((PetscObject)term), PETSC_ERR_PLIB, "failed to initialize mffd matrix");
  }
  PetscCall(PetscContainerGetPointer(container, (void **)&tp));
  PetscCheck(tp->term == term, PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_INCOMP, "Hessian shell matrix does not come from this TaoTerm");
  if (params) PetscCall(PetscObjectReference((PetscObject)params));
  PetscCall(VecDestroy(&tp->params));
  tp->params = params;
  PetscCall(MatMFFDSetBase(H, x, NULL));
  PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}
