
/*MC
     MATMFFD_WP - Implements an approach for computing the differencing parameter
        h used with the finite difference based matrix-free Jacobian.

      h = error_rel * sqrt(1 + ||U||) / ||a||

   Options Database Key:
.   -mat_mffd_compute_normu -Compute the norm of u every time see `MatMFFDWPSetComputeNormU()`

   Level: intermediate

   Notes:
   || U || does not change between linear iterations so is reused

   In `KSPGMRES` || a || == 1 and so does not need to ever be computed except at restart
    when it is recomputed.  Thus equires no global collectives when used with `KSPGMRES`

   Formula used:
     F'(u)*a = [F(u+h*a) - F(u)]/h where

   Reference:
.  * -  M. Pernice and H. F. Walker, "NITSOL: A Newton Iterative
      Solver for Nonlinear Systems", SIAM J. Sci. Stat. Comput.", 1998,
      vol 19, pp. 302--318.

.seealso: `MATMFFD`, `MATMFFD_DS`, `MatCreateMFFD()`, `MatCreateSNESMF()`, `MATMFFD_DS`
M*/

/*
    This include file defines the data structure  MatMFFD that
   includes information about the computation of h. It is shared by
   all implementations that people provide.

   See snesmfjdef.c for  a full set of comments on the routines below.
*/
#include <petsc/private/matimpl.h>
#include <../src/mat/impls/mffd/mffdimpl.h> /*I  "petscmat.h"   I*/

typedef struct {
  PetscReal normUfact; /* previous sqrt(1.0 + || U ||) */
  PetscBool computenormU;
} MatMFFD_WP;

/*
     MatMFFDCompute_WP - code for
   computing h with matrix-free finite differences.

  Input Parameters:
+   ctx - the matrix free context
.   U - the location at which you want the Jacobian
-   a - the direction you want the derivative

  Output Parameter:
.   h - the scale computed

*/
static PetscErrorCode MatMFFDCompute_WP(MatMFFD ctx, Vec U, Vec a, PetscScalar *h, PetscBool *zeroa)
{
  MatMFFD_WP *hctx = (MatMFFD_WP *)ctx->hctx;
  PetscReal   normU, norma;

  PetscFunctionBegin;
  if (!(ctx->count % ctx->recomputeperiod)) {
    if (hctx->computenormU || !ctx->ncurrenth) {
      PetscCall(VecNorm(U, NORM_2, &normU));
      hctx->normUfact = PetscSqrtReal(1.0 + normU);
    }
    PetscCall(VecNorm(a, NORM_2, &norma));
    if (norma == 0.0) {
      *zeroa = PETSC_TRUE;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    *zeroa = PETSC_FALSE;
    *h     = ctx->error_rel * hctx->normUfact / norma;
  } else {
    *h = ctx->currenth;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   MatMFFDView_WP - Prints information about this particular
     method for computing h. Note that this does not print the general
     information about the matrix free, that is printed by the calling
     routine.

  Input Parameters:
+   ctx - the matrix free context
-   viewer - the PETSc viewer

*/
static PetscErrorCode MatMFFDView_WP(MatMFFD ctx, PetscViewer viewer)
{
  MatMFFD_WP *hctx = (MatMFFD_WP *)ctx->hctx;
  PetscBool   iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    if (hctx->computenormU) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "    Computes normU\n"));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer, "    Does not compute normU\n"));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   MatMFFDSetFromOptions_WP - Looks in the options database for
     any options appropriate for this method

  Input Parameter:
.  ctx - the matrix free context

*/
static PetscErrorCode MatMFFDSetFromOptions_WP(MatMFFD ctx, PetscOptionItems *PetscOptionsObject)
{
  MatMFFD_WP *hctx = (MatMFFD_WP *)ctx->hctx;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "Walker-Pernice options");
  PetscCall(PetscOptionsBool("-mat_mffd_compute_normu", "Compute the norm of u", "MatMFFDWPSetComputeNormU", hctx->computenormU, &hctx->computenormU, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMFFDDestroy_WP(MatMFFD ctx)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectComposeFunction((PetscObject)ctx->mat, "MatMFFDWPSetComputeNormU_C", NULL));
  PetscCall(PetscFree(ctx->hctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMFFDWPSetComputeNormU_P(Mat mat, PetscBool flag)
{
  MatMFFD     ctx  = (MatMFFD)mat->data;
  MatMFFD_WP *hctx = (MatMFFD_WP *)ctx->hctx;

  PetscFunctionBegin;
  hctx->computenormU = flag;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    MatMFFDWPSetComputeNormU - Sets whether it computes the ||U|| used by the Walker-Pernice
             PETSc routine for computing h. With any Krylov solver this need only
             be computed during the first iteration and kept for later.

  Input Parameters:
+   A - the `MATMFFD` matrix
-   flag - `PETSC_TRUE` causes it to compute ||U||, `PETSC_FALSE` uses the previous value

  Options Database Key:
.   -mat_mffd_compute_normu <true,false> - true by default, false can save calculations but you
              must be sure that ||U|| has not changed in the mean time.

  Level: advanced

  Note:
   See the manual page for `MATMFFD_WP` for a complete description of the
   algorithm used to compute h.

.seealso: `MATMFFD_WP`, `MATMFFD`, `MatMFFDSetFunctionError()`, `MatCreateSNESMF()`
@*/
PetscErrorCode MatMFFDWPSetComputeNormU(Mat A, PetscBool flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscTryMethod(A, "MatMFFDWPSetComputeNormU_C", (Mat, PetscBool), (A, flag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
     MatCreateMFFD_WP - Standard PETSc code for
   computing h with matrix-free finite differences.

   Input Parameter:
.  ctx - the matrix free context created by MatCreateMFFD()

*/
PETSC_EXTERN PetscErrorCode MatCreateMFFD_WP(MatMFFD ctx)
{
  MatMFFD_WP *hctx;

  PetscFunctionBegin;
  /* allocate my own private data structure */
  PetscCall(PetscNew(&hctx));
  ctx->hctx          = (void *)hctx;
  hctx->computenormU = PETSC_FALSE;

  /* set the functions I am providing */
  ctx->ops->compute        = MatMFFDCompute_WP;
  ctx->ops->destroy        = MatMFFDDestroy_WP;
  ctx->ops->view           = MatMFFDView_WP;
  ctx->ops->setfromoptions = MatMFFDSetFromOptions_WP;

  PetscCall(PetscObjectComposeFunction((PetscObject)ctx->mat, "MatMFFDWPSetComputeNormU_C", MatMFFDWPSetComputeNormU_P));
  PetscFunctionReturn(PETSC_SUCCESS);
}
